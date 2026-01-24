#include "inference_node.hpp"

void InferenceNode::setup_model(std::unique_ptr<ModelContext>& ctx, std::string model_path, int input_size){
    if (!ctx) {
        ctx = std::make_unique<ModelContext>();
    }

    Ort::SessionOptions session_options;
    session_options.DisablePerSessionThreads();
    session_options.EnableCpuMemArena();
    session_options.EnableMemPattern();
    session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
    
    ctx->session = std::make_unique<Ort::Session>(*env_, model_path.c_str(), session_options);
    
    ctx->num_inputs = ctx->session->GetInputCount();
    ctx->input_names.resize(ctx->num_inputs);
    ctx->input_buffer.resize(input_size);

    for (size_t i = 0; i < ctx->num_inputs; i++) {
        Ort::AllocatedStringPtr input_name = ctx->session->GetInputNameAllocated(i, allocator_);
        ctx->input_names[i] = input_name.get();
        auto type_info = ctx->session->GetInputTypeInfo(i);
        ctx->input_shape = type_info.GetTensorTypeAndShapeInfo().GetShape();
        if (ctx->input_shape[0] == -1) ctx->input_shape[0] = 1;
    }

    ctx->num_outputs = ctx->session->GetOutputCount();
    ctx->output_names.resize(ctx->num_outputs);
    ctx->output_buffer.resize(joint_num_);

    for (size_t i = 0; i < ctx->num_outputs; i++) {
        Ort::AllocatedStringPtr output_name = ctx->session->GetOutputNameAllocated(i, allocator_);
        ctx->output_names[i] = output_name.get();
        auto type_info = ctx->session->GetOutputTypeInfo(i);
        ctx->output_shape = type_info.GetTensorTypeAndShapeInfo().GetShape();
    }

    ctx->input_names_raw = std::vector<const char *>(ctx->num_inputs, nullptr);
    ctx->output_names_raw = std::vector<const char *>(ctx->num_outputs, nullptr);
    for (size_t i = 0; i < ctx->num_inputs; i++) {
        ctx->input_names_raw[i] = ctx->input_names[i].c_str();
    }
    for (size_t i = 0; i < ctx->num_outputs; i++) {
        ctx->output_names_raw[i] = ctx->output_names[i].c_str();
    }

    ctx->memory_info = std::make_unique<Ort::MemoryInfo>(Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU));
    
    ctx->input_tensor = std::make_unique<Ort::Value>(Ort::Value::CreateTensor<float>(
        *ctx->memory_info, ctx->input_buffer.data(), ctx->input_buffer.size(), ctx->input_shape.data(), ctx->input_shape.size()));
        
    ctx->output_tensor = std::make_unique<Ort::Value>(Ort::Value::CreateTensor<float>(
        *ctx->memory_info, ctx->output_buffer.data(), ctx->output_buffer.size(), ctx->output_shape.data(), ctx->output_shape.size()));
}

void InferenceNode::setup_motors(){
    size_t count = 0;
    motors_.resize(joint_num_);
    for (size_t i = 0; i < motor_interface_.size(); ++i){
        for (size_t j = 0; j < motor_num_[i]; ++j){
            motors_[count] = MotorDriver::create_motor(motor_id_[count], motor_interface_type_, motor_interface_[i], motor_type_, motor_model_[count], master_id_offset_);
            count += 1;
        }
    }
}

void InferenceNode::setup_imu(){
    imu_ = IMUDriver::create_imu(imu_id_, imu_interface_type_, imu_interface_, imu_type_, baudrate_);
}

void InferenceNode::reset() {
    is_running_.store(false);
    std::fill(obs_.begin(), obs_.end(), 0.0f);
    std::fill(joint_obs_.begin(), joint_obs_.end(), 0.0f);
    std::fill(motion_pos_.begin(), motion_pos_.end(), 0.0f);
    std::fill(motion_vel_.begin(), motion_vel_.end(), 0.0f);
    std::fill(cmd_vel_.begin(), cmd_vel_.end(), 0.0f);
    std::fill(quat_.begin(), quat_.end(), 0.0f);
    std::fill(ang_vel_.begin(), ang_vel_.end(), 0.0f);
    if (active_ctx_) {
        std::fill(active_ctx_->input_buffer.begin(), active_ctx_->input_buffer.end(), 0.0f);
        std::fill(active_ctx_->output_buffer.begin(), active_ctx_->output_buffer.end(), 0.0f);
    }
    std::fill(act_.begin(), act_.end(), 0.0f);
    std::fill(last_act_.begin(), last_act_.end(), 0.0f);
    std::fill(joint_torques_.begin(), joint_torques_.end(), 0.0f);
    is_first_frame_ = true;
    motion_frame_ = 0;
    is_interrupt_.store(false);
    is_beyondmimic_.store(false);
    if(use_interrupt_){
        std::fill(interrupt_action_.begin(), interrupt_action_.end(), 0.0f);
    }
    if(use_attn_enc_){
        std::fill(perception_obs_.begin(), perception_obs_.end(), 0.0f);
    }
}

void InferenceNode::apply_action() {
    if(!is_running_.load() || !is_init_.load()){
        return;
    }

    std::vector<float> motor_target;
    {
        std::unique_lock<std::mutex> lock(act_mutex_);
        for (size_t i = 0; i < act_.size(); i++) {
            act_[i] = act_alpha_ * act_[i] + (1 - act_alpha_) * last_act_[i];
        }
        last_act_ = act_;
        motor_target = act_;
    }

    if(use_interrupt_ && is_interrupt_.load()){
        std::unique_lock<std::mutex> lock(interrupt_mutex_);
        for (size_t i = 0; i < 10; i++) {
            motor_target[14 + i] = interrupt_action_[i];
        }
    }

    {
        std::unique_lock<std::mutex> lock(joint_mutex_);
        std::vector<std::function<void()>> tasks;
        size_t count = 0;
        exec_motors_parallel([this](std::shared_ptr<MotorDriver>& motor, int idx) {
            joint_obs_[idx] = motor->get_motor_pos() * motor_sign_[idx];
            joint_obs_[joint_num_ + idx] = motor->get_motor_spd() * motor_sign_[idx];
            joint_torques_[idx] = motor->get_motor_current() * motor_sign_[idx];
            if (motor->get_response_count() > offline_threshold_) {
                RCLCPP_FATAL(this->get_logger(), "Motor ID %d is offline! Shutting down...", idx);
                rclcpp::shutdown();
            }
        });

        if (!close_chain_motor_id_.empty()){
            process_close_chain(joint_obs_.data(), joint_obs_.data() + joint_num_, joint_torques_.data(), motor_target.data(), false);
        }
    }

    exec_motors_parallel([this, &motor_target](std::shared_ptr<MotorDriver>& motor, int idx) {
        if (std::find(close_chain_motor_id_.begin(), close_chain_motor_id_.end(), idx) == close_chain_motor_id_.end()){
            motor->motor_mit_cmd(motor_target[idx] * motor_sign_[idx], 0.0f, kp_[idx], kd_[idx], 0.0f);
        } else {
            motor->motor_mit_cmd(0.0f, 0.0f, 0.0f, 0.0f, motor_target[idx] * motor_sign_[idx]);
        }
        std::this_thread::sleep_for(std::chrono::microseconds(200));
    });
}

void InferenceNode::inference() {
    pthread_setname_np(pthread_self(), "inference");
    struct sched_param sp{}; sp.sched_priority = 65;
    pthread_setschedparam(pthread_self(), SCHED_FIFO, &sp);
    auto period = std::chrono::microseconds(static_cast<long long>(dt_ * 1000 * 1000 * decimation_));

    while(rclcpp::ok()){
        auto loop_start = std::chrono::steady_clock::now();
        if(!is_running_.load()){
            std::this_thread::sleep_for(period);
            continue;
        }

        int offset = 0;

        if(is_beyondmimic_.load()){
            motion_pos_ = motion_loader_->get_pos(motion_frame_);
            motion_vel_ = motion_loader_->get_vel(motion_frame_);
            motion_frame_ += 1;
            if(motion_frame_ >= motion_loader_->get_num_frames()){
                motion_frame_ = motion_loader_->get_num_frames() - 1;
            }
            for(int i = 0; i < joint_num_; i++){
                obs_[i + offset] = motion_pos_[i];
                obs_[i + joint_num_ + offset] = motion_vel_[i];
            }
            offset += joint_num_ * 2;
        }

        ang_vel_ = imu_->get_ang_vel();
        for (int i = 0; i < 3; i++) {
            obs_[i + offset] = ang_vel_[i] * obs_scales_ang_vel_;
        }
        offset += 3;
        quat_ = imu_->get_quat();
        Eigen::Quaternionf q_b2w(quat_[0], quat_[1], quat_[2], quat_[3]);
        Eigen::Vector3f gravity_w(0.0f, 0.0f, -1.0f);
        Eigen::Quaternionf q_w2b = q_b2w.inverse();
        Eigen::Vector3f gravity_b = q_w2b * gravity_w;
        if (gravity_b.z() > gravity_z_upper_){
            RCLCPP_FATAL(this->get_logger(), "Robot fell down! Shutting down...");
            rclcpp::shutdown();
            return;
        }
        obs_[0 + offset] = gravity_b.x() * obs_scales_gravity_b_;
        obs_[1 + offset] = gravity_b.y() * obs_scales_gravity_b_;
        obs_[2 + offset] = gravity_b.z() * obs_scales_gravity_b_;
        offset += 3;
        publish_imu();

        if (!is_beyondmimic_.load()){
            std::unique_lock<std::mutex> lock(cmd_mutex_);
            obs_[0 + offset] = cmd_vel_[0] * obs_scales_lin_vel_;
            obs_[1 + offset] = cmd_vel_[1] * obs_scales_lin_vel_;
            obs_[2 + offset] = cmd_vel_[2] * obs_scales_ang_vel_;
            offset += 3;
        }

        {
            std::unique_lock<std::mutex> lock(joint_mutex_);
            for (int i = 0; i < joint_num_; i++) {
                obs_[offset + i] = joint_obs_[usd2urdf_[i]] * obs_scales_dof_pos_;
                obs_[offset + joint_num_ + i] = joint_obs_[joint_num_ + usd2urdf_[i]] * obs_scales_dof_vel_;
            }
            publish_joint_states();
        }
        offset += joint_num_ * 2;

        for (int i = 0; i < joint_num_; i++) {
            obs_[offset + i] = active_ctx_->output_buffer[i];
        }
        offset += joint_num_;

        if (use_interrupt_){
            obs_[offset] = is_interrupt_.load() ? 1.0 : 0.0;
            offset += 1;
        }

        std::transform(obs_.begin(), obs_.end(), obs_.begin(), [this](float val) {
            return std::clamp(val, -clip_observations_, clip_observations_);
        });


        bool is_beyondmimic = is_beyondmimic_.load();
        int obs_num = is_beyondmimic ? motion_obs_num_: obs_num_;
        int frame_stack = is_beyondmimic ? motion_frame_stack_ : frame_stack_;
        if (is_first_frame_) {
            for (int i = 0; i < frame_stack; i++) {
                std::copy(obs_.begin(), obs_.end(), active_ctx_->input_buffer.begin() + i * obs_num);
            }
            if(use_attn_enc_){
                std::unique_lock<std::mutex> lock(perception_mutex_);
                std::copy(perception_obs_.begin(), perception_obs_.end(), active_ctx_->input_buffer.begin() + frame_stack * obs_num);
            }
            is_first_frame_ = false;
        } else {
            std::copy(active_ctx_->input_buffer.begin() + obs_num, active_ctx_->input_buffer.begin() + frame_stack * obs_num, active_ctx_->input_buffer.begin());
            std::copy(obs_.begin(), obs_.end(), active_ctx_->input_buffer.begin() + (frame_stack - 1) * obs_num);
            if(use_attn_enc_){
                std::unique_lock<std::mutex> lock(perception_mutex_);
                std::copy(perception_obs_.begin(), perception_obs_.end(), active_ctx_->input_buffer.begin() + frame_stack * obs_num);
            }
        }

        active_ctx_->session->Run(Ort::RunOptions{nullptr}, 
            active_ctx_->input_names_raw.data(), active_ctx_->input_tensor.get(), active_ctx_->num_inputs,
            active_ctx_->output_names_raw.data(), active_ctx_->output_tensor.get(), active_ctx_->num_outputs);
        
        {
            std::unique_lock<std::mutex> lock(act_mutex_);
            for (int i = 0; i < active_ctx_->output_buffer.size(); i++) {
                active_ctx_->output_buffer[i] = std::clamp(active_ctx_->output_buffer[i], -clip_actions_, clip_actions_);
                act_[usd2urdf_[i]] = active_ctx_->output_buffer[i];
                act_[usd2urdf_[i]] = act_[usd2urdf_[i]] * action_scale_ + joint_default_angle_[usd2urdf_[i]];
            }
            publish_action();
        }

        auto loop_end = std::chrono::steady_clock::now();
        // 使用微秒进行计算
        auto elapsed_time = std::chrono::duration_cast<std::chrono::microseconds>(loop_end - loop_start);
        auto sleep_time = period - elapsed_time;

        if (sleep_time > std::chrono::microseconds(0)) {
            std::this_thread::sleep_for(sleep_time);
        } else {
            // 警告信息也使用更精确的单位
            RCLCPP_WARN(this->get_logger(), "Inference loop overran! Took %ld us, but period is %ld us.", elapsed_time.count(), period.count());
        }
    }
}

void InferenceNode::reset_motors() {
    size_t count = 0;
    std::vector<float> motor_default_q(joint_default_angle_.begin(), joint_default_angle_.end());

    if (!close_chain_motor_id_.empty()){
        std::vector<float> dummy_vel(joint_num_, 0.0f);
        std::vector<float> dummy_tau(joint_num_, 0.0f);
        std::vector<float> dummy_target(joint_num_, 0.0f);
        process_close_chain(motor_default_q.data(), dummy_vel.data(), dummy_tau.data(), dummy_target.data(), true);
    }

    exec_motors_parallel([this, &motor_default_q](std::shared_ptr<MotorDriver>& motor, int idx) {
        if (std::find(close_chain_motor_id_.begin(), close_chain_motor_id_.end(), idx) == close_chain_motor_id_.end()) {
                motor->motor_mit_cmd(motor_default_q[idx] * motor_sign_[idx], 0.0f, kp_[idx]/2.0f, kd_[idx]/2.0f, 0.0f);
        } else {
                motor->motor_mit_cmd(motor_default_q[idx] * motor_sign_[idx], 0.0f, 0.0f, 0.0f, 0.0f);
        }
        std::this_thread::sleep_for(std::chrono::microseconds(200));
    });
}

void InferenceNode::read_motors() {
    {
        std::unique_lock<std::mutex> lock(joint_mutex_);
        exec_motors_parallel([this](std::shared_ptr<MotorDriver>& motor, int idx) {
            motor->refresh_motor_status();
            std::this_thread::sleep_for(std::chrono::microseconds(200));
            joint_obs_[idx] = motor->get_motor_pos() * motor_sign_[idx];
            joint_obs_[joint_num_ + idx] = motor->get_motor_spd() * motor_sign_[idx];
            joint_torques_[idx] = motor->get_motor_current() * motor_sign_[idx];
        });

        if (!close_chain_motor_id_.empty()) {
             std::vector<float> dummy_target(joint_num_, 0.0f);
             process_close_chain(joint_obs_.data(), joint_obs_.data() + joint_num_, joint_torques_.data(), dummy_target.data(), true);
        }
        publish_joint_states();
    }
}

void InferenceNode::set_zeros() {
    exec_motors_parallel([](std::shared_ptr<MotorDriver>& motor, int idx) {
        motor->set_motor_zero();
        std::this_thread::sleep_for(std::chrono::microseconds(200));
    });
}

void InferenceNode::clear_errors() {
    exec_motors_parallel([](std::shared_ptr<MotorDriver>& motor, int idx) {
        motor->clear_motor_error();
        std::this_thread::sleep_for(std::chrono::microseconds(200));
    });
}

void InferenceNode::init_motors() {
    exec_motors_parallel([](std::shared_ptr<MotorDriver>& motor, int idx) {
        motor->init_motor();
        std::this_thread::sleep_for(std::chrono::microseconds(200));
    });
    is_init_.store(true);
}

void InferenceNode::deinit_motors() {
    exec_motors_parallel([](std::shared_ptr<MotorDriver>& motor, int idx) {
        motor->deinit_motor();
        std::this_thread::sleep_for(std::chrono::microseconds(200));
    });
    is_init_.store(false);
}

void InferenceNode::exec_motors_parallel(std::function<void(std::shared_ptr<MotorDriver>&, int)> cmd_func) {
    std::unique_lock<std::mutex> lock(motors_mutex_);
    std::vector<std::function<void()>> tasks;
    size_t count = 0;
    
    for (size_t i = 0; i < motor_interface_.size(); ++i) {
        size_t num_motors = motor_num_[i];
        size_t start_idx = count;
        tasks.push_back([this, start_idx, num_motors, cmd_func]() {
            for (size_t j = 0; j < num_motors; ++j) {
                size_t idx = start_idx + j;
                cmd_func(motors_[idx], idx); 
            }
        });
        count += num_motors;
    }
    thread_pool_->run_parallel(tasks);
}



void InferenceNode::process_close_chain(float* q_in, float* vel_in, float* tau_in, float* target, bool forward_only) {
    Eigen::VectorXd q(2), vel(2), tau(2);
    int idx1 = close_chain_motor_id_[0];
    int idx2 = close_chain_motor_id_[1];

    q << q_in[idx1], q_in[idx2];
    vel << vel_in[idx1], vel_in[idx2];
    tau << tau[idx1], tau_in[idx2];

    ankle_decouple_->get_forwardQVT(q, vel, tau, true);
    q_in[idx1] = q[0];
    q_in[idx2] = q[1];
    vel_in[idx1] = vel[0];
    vel_in[idx2] = vel[1];
    tau_in[idx1] = tau[0];
    tau_in[idx2] = tau[1];

    if (!forward_only) {
        tau << kp_[idx1] * (q[0] - target[idx1]) + kd_[idx1] * (0.0f - vel[0]),
               kp_[idx2] * (q[1] - target[idx2]) + kd_[idx2] * (0.0f - vel[1]);
        ankle_decouple_->get_decoupleQVT(q, vel, tau, true);
        target[idx1] = tau[0];
        target[idx2] = tau[1];
    }

    idx1 = close_chain_motor_id_[2];
    idx2 = close_chain_motor_id_[3];
    q << q_in[idx1], q_in[idx2];
    vel << vel_in[idx1], vel_in[idx2];
    tau << tau_in[idx1], tau_in[idx2];

    ankle_decouple_->get_forwardQVT(q, vel, tau, false);
    q_in[idx1] = q[0];
    q_in[idx2] = q[1];
    vel_in[idx1] = vel[0];
    vel_in[idx2] = vel[1];
    tau_in[idx1] = tau[0];
    tau_in[idx2] = tau[1];

    if (!forward_only) {
        tau << kp_[idx1] * (q[0] - target[idx1]) + kd_[idx1] * (0.0f - vel[0]),
               kp_[idx2] * (q[1] - target[idx2]) + kd_[idx2] * (0.0f - vel[1]);
        ankle_decouple_->get_decoupleQVT(q, vel, tau, false);
        target[idx1] = tau[0];
        target[idx2] = tau[1];
    }
}

int main(int argc, char **argv) {
    rclcpp::init(argc, argv);
    auto node = std::make_shared<InferenceNode>();
    rclcpp::executors::MultiThreadedExecutor executor(rclcpp::ExecutorOptions(), 4);
    executor.add_node(node);
    RCLCPP_INFO(node->get_logger(), "Press 'A' to initialize/deinitialize motors");
    RCLCPP_INFO(node->get_logger(), "Press 'X' to reset motors");
    RCLCPP_INFO(node->get_logger(), "Press 'B' to start/pause inference");
    RCLCPP_INFO(node->get_logger(), "Press 'Y' to switch between joystick and /cmd_vel control");
    executor.spin();
    rclcpp::shutdown();
    return 0;
}
