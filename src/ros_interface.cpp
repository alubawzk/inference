#include "inference_node.hpp"

void InferenceNode::load_config() {
    this->declare_parameter<std::string>("model_name", "1.onnx");
    this->declare_parameter<std::string>("motion_name", "motion.npz");
    this->declare_parameter<std::string>("motion_model_name", "1.onnx");
    this->declare_parameter<float>("act_alpha", 0.9);
    this->declare_parameter<float>("gyro_alpha", 0.9);
    this->declare_parameter<float>("angle_alpha", 0.9);
    this->declare_parameter<int>("intra_threads", -1);
    this->declare_parameter<bool>("use_interrupt", false);
    this->declare_parameter<bool>("use_beyondmimic", false);
    this->declare_parameter<bool>("use_attn_enc", false);
    this->declare_parameter<int>("obs_num", 78);
    this->declare_parameter<int>("motion_obs_num", 121);
    this->declare_parameter<int>("perception_obs_num", 187);
    this->declare_parameter<std::string>("perception_obs_topic", "elevation_data");
    this->declare_parameter<int>("frame_stack", 15);
    this->declare_parameter<int>("motion_frame_stack", 1);
    this->declare_parameter<int>("joint_num", 23);
    this->declare_parameter<int>("decimation", 10);
    this->declare_parameter<float>("dt", 0.001);
    this->declare_parameter<float>("obs_scales_lin_vel", 1.0);
    this->declare_parameter<float>("obs_scales_ang_vel", 1.0);
    this->declare_parameter<float>("obs_scales_dof_pos", 1.0);
    this->declare_parameter<float>("obs_scales_dof_vel", 1.0);
    this->declare_parameter<float>("obs_scales_gravity_b", 1.0);
    this->declare_parameter<float>("clip_observations", 100.0);
    this->declare_parameter<float>("action_scale", 0.3);
    this->declare_parameter<float>("clip_actions", 18.0);
    this->declare_parameter<bool>("use_sine_trajectory", false);
    this->declare_parameter<float>("sine_freq_hz", 500.0);
    this->declare_parameter<float>("sine_amplitude", 1.0);
    this->declare_parameter<int>("joy_button_init_motors", 0);
    this->declare_parameter<int>("joy_button_reset_motors", 2);
    this->declare_parameter<int>("joy_button_toggle_inference", 1);
    this->declare_parameter<int>("joy_button_toggle_control", 3);
    this->declare_parameter<int>("joy_button_toggle_special_mode", 4);
    this->declare_parameter<std::vector<long int>>("usd2urdf", std::vector<long int>{});
    this->declare_parameter<std::vector<double>>("clip_cmd", std::vector<double>{});
    this->declare_parameter<std::vector<double>>("joint_default_angle", std::vector<double>{});
    this->declare_parameter<std::vector<double>>("joint_limits", std::vector<double>{});


    this->get_parameter("model_name", model_name_);
    this->get_parameter("motion_name", motion_name_);
    this->get_parameter("motion_model_name", motion_model_name_);
    this->get_parameter("act_alpha", act_alpha_);
    this->get_parameter("gyro_alpha", gyro_alpha_);
    this->get_parameter("angle_alpha", angle_alpha_);
    this->get_parameter("intra_threads", intra_threads_);
    this->get_parameter("use_interrupt", use_interrupt_);
    this->get_parameter("use_beyondmimic", use_beyondmimic_);
    this->get_parameter("use_attn_enc", use_attn_enc_);
    this->get_parameter("obs_num", obs_num_);
    this->get_parameter("motion_obs_num", motion_obs_num_);
    this->get_parameter("perception_obs_num", perception_obs_num_);
    this->get_parameter("perception_obs_topic", perception_obs_topic_);
    this->get_parameter("frame_stack", frame_stack_);
    this->get_parameter("motion_frame_stack", motion_frame_stack_);
    this->get_parameter("joint_num", joint_num_);
    this->get_parameter("decimation", decimation_);
    this->get_parameter("dt", dt_);
    this->get_parameter("obs_scales_lin_vel", obs_scales_lin_vel_);
    this->get_parameter("obs_scales_ang_vel", obs_scales_ang_vel_);
    this->get_parameter("obs_scales_dof_pos", obs_scales_dof_pos_);
    this->get_parameter("obs_scales_dof_vel", obs_scales_dof_vel_);
    this->get_parameter("obs_scales_gravity_b", obs_scales_gravity_b_);
    this->get_parameter("clip_observations", clip_observations_);
    this->get_parameter("action_scale", action_scale_);
    this->get_parameter("clip_actions", clip_actions_);
    this->get_parameter("use_sine_trajectory", use_sine_trajectory_);
    this->get_parameter("sine_freq_hz", sine_freq_hz_);
    this->get_parameter("sine_amplitude", sine_amplitude_);
    this->get_parameter("joy_button_init_motors", joy_btn_init_);
    this->get_parameter("joy_button_reset_motors", joy_btn_reset_);
    this->get_parameter("joy_button_toggle_inference", joy_btn_toggle_inference_);
    this->get_parameter("joy_button_toggle_control", joy_btn_toggle_control_);
    this->get_parameter("joy_button_toggle_special_mode", joy_btn_toggle_special_mode_);
    this->get_parameter("usd2urdf", usd2urdf_);
    this->get_parameter("clip_cmd", clip_cmd_);
    this->get_parameter("joint_default_angle", joint_default_angle_);
    this->get_parameter("joint_limits", joint_limits_);


    model_path_ = std::string(ROOT_DIR) + "models/" + model_name_;
    motion_path_ = std::string(ROOT_DIR) + "motions/" + motion_name_;
    motion_model_path_ = std::string(ROOT_DIR) + "models/" + motion_model_name_;
    RCLCPP_INFO(this->get_logger(), "model_path: %s", model_path_.c_str());
    RCLCPP_INFO(this->get_logger(), "motion_path: %s", motion_path_.c_str());
    RCLCPP_INFO(this->get_logger(), "motion_model_path: %s", motion_model_path_.c_str());
    RCLCPP_INFO(this->get_logger(), "act_alpha: %f", act_alpha_);
    RCLCPP_INFO(this->get_logger(), "gyro_alpha: %f", gyro_alpha_);
    RCLCPP_INFO(this->get_logger(), "angle_alpha: %f", angle_alpha_);
    RCLCPP_INFO(this->get_logger(), "intra_threads: %d", intra_threads_);
    RCLCPP_INFO(this->get_logger(), "use_interrupt: %s", use_interrupt_ ? "true" : "false");
    RCLCPP_INFO(this->get_logger(), "use_beyondmimic: %s", use_beyondmimic_ ? "true" : "false");
    RCLCPP_INFO(this->get_logger(), "obs_num: %d", obs_num_);
    RCLCPP_INFO(this->get_logger(), "perception_obs_num: %d", perception_obs_num_);
    RCLCPP_INFO(this->get_logger(), "perception_obs_topic: %s", perception_obs_topic_.c_str());
    RCLCPP_INFO(this->get_logger(), "joint_num: %d", joint_num_);
    RCLCPP_INFO(this->get_logger(), "decimation: %d", decimation_);
    RCLCPP_INFO(this->get_logger(), "dt: %f", dt_);
    RCLCPP_INFO(this->get_logger(), "obs_scales_lin_vel: %f", obs_scales_lin_vel_);
    RCLCPP_INFO(this->get_logger(), "obs_scales_ang_vel: %f", obs_scales_ang_vel_);
    RCLCPP_INFO(this->get_logger(), "obs_scales_dof_pos: %f", obs_scales_dof_pos_);
    RCLCPP_INFO(this->get_logger(), "obs_scales_dof_vel: %f", obs_scales_dof_vel_);
    RCLCPP_INFO(this->get_logger(), "obs_scales_gravity_b: %f", obs_scales_gravity_b_);
    RCLCPP_INFO(this->get_logger(), "action_scale: %f", action_scale_);
    RCLCPP_INFO(this->get_logger(), "clip_actions: %f", clip_actions_);
    RCLCPP_INFO(this->get_logger(), "use_sine_trajectory: %s", use_sine_trajectory_ ? "true" : "false");
    RCLCPP_INFO(this->get_logger(), "sine_freq_hz: %f", sine_freq_hz_);
    RCLCPP_INFO(this->get_logger(), "sine_amplitude: %f", sine_amplitude_);
    RCLCPP_INFO(this->get_logger(), "joy_button_init_motors: %d", joy_btn_init_);
    RCLCPP_INFO(this->get_logger(), "joy_button_reset_motors: %d", joy_btn_reset_);
    RCLCPP_INFO(this->get_logger(), "joy_button_toggle_inference: %d", joy_btn_toggle_inference_);
    RCLCPP_INFO(this->get_logger(), "joy_button_toggle_control: %d", joy_btn_toggle_control_);
    RCLCPP_INFO(this->get_logger(), "joy_button_toggle_special_mode: %d", joy_btn_toggle_special_mode_);
    print_vector<long int>("usd2urdf", usd2urdf_);
    print_vector<double>("clip_cmd", clip_cmd_);
    print_vector<double>("joint_default_angle", joint_default_angle_);
    print_vector<double>("joint_limits", joint_limits_);
}

void InferenceNode::subs_joy_callback(const std::shared_ptr<sensor_msgs::msg::Joy> msg) {
    auto get_button_value = [&](int idx) -> int {
        if (idx < 0) {
            return 0;
        }
        size_t button_idx = static_cast<size_t>(idx);
        if (button_idx >= msg->buttons.size()) {
            return 0;
        }
        return msg->buttons[button_idx];
    };
    auto get_axis_value = [&](size_t idx) -> float {
        if (idx >= msg->axes.size()) {
            return 0.0f;
        }
        return msg->axes[idx];
    };
    auto is_rising_edge = [&](int button_idx, int &last_state) -> bool {
        const int current_state = get_button_value(button_idx);
        const bool rising_edge = (current_state == 1 && current_state != last_state);
        last_state = current_state;
        return rising_edge;
    };

    if (is_joy_control_){
        std::unique_lock<std::mutex> lock(cmd_mutex_);
        cmd_vel_[0] = std::clamp(get_axis_value(3) * clip_cmd_[1], clip_cmd_[0], clip_cmd_[1]);
        cmd_vel_[1] = std::clamp(get_axis_value(2) * clip_cmd_[3], clip_cmd_[2], clip_cmd_[3]);
            if (get_button_value(6) == 1) {
            cmd_vel_[2] = std::clamp(static_cast<float>(get_button_value(6)) * clip_cmd_[5], clip_cmd_[4], clip_cmd_[5]);
            } else if (get_button_value(7) == 1) {
            cmd_vel_[2] = std::clamp(-static_cast<float>(get_button_value(7)) * clip_cmd_[5], clip_cmd_[4], clip_cmd_[5]);
            } else {
            cmd_vel_[2] = 0.0;
        }
    }
    if (is_rising_edge(joy_btn_init_, last_joy_btn_init_)) {
        if(is_running_.load()){
            reset();
            RCLCPP_INFO(this->get_logger(), "Inference paused");
        }
        if (robot_->is_init_.load()){
            robot_->deinit_motors();
            RCLCPP_INFO(this->get_logger(), "Motors deinitialized");
        } else {
            robot_->init_motors();
            RCLCPP_INFO(this->get_logger(), "Motors initialized");
        }
    }
    if (is_rising_edge(joy_btn_reset_, last_joy_btn_reset_)) {
        if (is_running_.load()){
            reset();
            RCLCPP_INFO(this->get_logger(), "Inference paused");
        }
        if (!robot_->is_init_.load()){
            RCLCPP_WARN(this->get_logger(), "Motors are not initialized!");
        } else {
            robot_->reset_joints(joint_default_angle_);
            RCLCPP_INFO(this->get_logger(), "Motors reset");
        }
    }
    if (is_rising_edge(joy_btn_toggle_inference_, last_joy_btn_toggle_inference_)) {
        is_running_.store(!is_running_.load());
        RCLCPP_INFO(this->get_logger(), "Inference %s", is_running_.load() ? "started" : "paused");
    }
    if (is_rising_edge(joy_btn_toggle_control_, last_joy_btn_toggle_control_)) {
        is_joy_control_.store(!is_joy_control_);
        RCLCPP_INFO(this->get_logger(), "Controlled by %s", is_joy_control_.load() ? "joy" : "/cmd_vel");
    }
    if (use_interrupt_ || use_beyondmimic_) {
        if (is_rising_edge(joy_btn_toggle_special_mode_, last_joy_btn_toggle_special_mode_)) {
            if(use_interrupt_){
                is_interrupt_.store(!is_interrupt_.load());
                RCLCPP_INFO(this->get_logger(), "Interrupt mode %s", is_interrupt_.load() ? "enabled" : "disabled");
            }
            if(use_beyondmimic_){
                is_running_.store(false);
                std::this_thread::sleep_for(std::chrono::milliseconds(50));
                is_beyondmimic_.store(!is_beyondmimic_.load());
                bool is_beyondmimic = is_beyondmimic_.load();
                active_ctx_ = is_beyondmimic ? motion_ctx_.get() : normal_ctx_.get();
                int obs_num = is_beyondmimic ? motion_obs_num_ : obs_num_;
                obs_.resize(obs_num);
                std::fill(obs_.begin(), obs_.end(), 0.0f);
                std::fill(joint_pos_.begin(), joint_pos_.end(), 0.0f);
                std::fill(joint_vel_.begin(), joint_vel_.end(), 0.0f);
                std::fill(motion_pos_.begin(), motion_pos_.end(), 0.0f);
                std::fill(motion_vel_.begin(), motion_vel_.end(), 0.0f);
                std::fill(cmd_vel_.begin(), cmd_vel_.end(), 0.0f);
                std::fill(quat_.begin(), quat_.end(), 0.0f);
                std::fill(ang_vel_.begin(), ang_vel_.end(), 0.0f);
                std::fill(active_ctx_->input_buffer.begin(), active_ctx_->input_buffer.end(), 0.0f);
                std::fill(active_ctx_->output_buffer.begin(), active_ctx_->output_buffer.end(), 0.0f);
                std::fill(act_.begin(), act_.end(), 0.0f);
                std::fill(last_act_.begin(), last_act_.end(), 0.0f);
                is_first_frame_ = true;
                motion_frame_ = 0;
                std::this_thread::sleep_for(std::chrono::milliseconds(50));
                is_running_.store(true);
                RCLCPP_INFO(this->get_logger(), "Beyondmimic mode %s", is_beyondmimic ? "enabled" : "disabled");
            }
        }
    }
}

void InferenceNode::subs_cmd_callback(const std::shared_ptr<geometry_msgs::msg::Twist> msg){
    if(!is_joy_control_){
        std::unique_lock<std::mutex> lock(cmd_mutex_);
        cmd_vel_[0] = std::clamp(msg->linear.x, clip_cmd_[0], clip_cmd_[1]);
        cmd_vel_[1] = std::clamp(msg->linear.y, clip_cmd_[2], clip_cmd_[3]);
        cmd_vel_[2] = std::clamp(msg->angular.z, clip_cmd_[4], clip_cmd_[5]);
    }
}

void InferenceNode::subs_elevation_callback(const std::shared_ptr<std_msgs::msg::Float32MultiArray> msg){
    if(use_attn_enc_){
        std::unique_lock<std::mutex> lock(perception_mutex_);
        for(int i = 0; i < perception_obs_num_; i++){
            perception_obs_[i] = msg->data[i];
        }
    }
}

void InferenceNode::subs_joint_state_callback(const std::shared_ptr<sensor_msgs::msg::JointState> msg){
    if(use_interrupt_ && is_interrupt_.load()){
        std::unique_lock<std::mutex> lock(interrupt_mutex_);
        for(size_t i = 0; i < 10; i++){
            interrupt_action_[i] = msg->position[i];
        }
    }
}

void InferenceNode::reset_joints_srv(const std::shared_ptr<std_srvs::srv::Trigger::Request> request,
                                     std::shared_ptr<std_srvs::srv::Trigger::Response> response) {
    if (is_running_.load()) {
        response->success = false;
        response->message = "Inference is running, cannot reset joints.";
        return;
    }
    if (!robot_->is_init_.load()) {
        response->success = false;
        response->message = "Motors are not initialized, cannot reset joints.";
        return;
    }
    try {
        robot_->reset_joints(joint_default_angle_);
        response->success = true;
        response->message = "Joints reset successfully";
    } catch (const std::exception& e) {
        response->success = false;
        response->message = e.what();
    }
}

void InferenceNode::refresh_joints_srv(const std::shared_ptr<std_srvs::srv::Trigger::Request> request,
                                     std::shared_ptr<std_srvs::srv::Trigger::Response> response) {
    if (!robot_->is_init_.load()) {
        response->success = false;
        response->message = "Motors are not initialized, cannot refresh motors.";
        return;
    }
    try {
        robot_->refresh_joints();
        response->success = true;
        response->message = "Motors refresh successfully";
    } catch (const std::exception& e) {
        response->success = false;
        response->message = e.what();
    }
}

void InferenceNode::read_joints_srv(const std::shared_ptr<std_srvs::srv::Trigger::Request> request,
                                     std::shared_ptr<std_srvs::srv::Trigger::Response> response) {
    if (!robot_->is_init_.load()) {
        response->success = false;
        response->message = "Motors are not initialized, cannot read joints.";
        return;
    }
    try {
        response->success = true;
        response->message = "Joints read successfully";
        read_joints();
        publish_joint_states();
    } catch (const std::exception& e) {
        response->success = false;
        response->message = e.what();
    }
}

void InferenceNode::read_imu_srv(const std::shared_ptr<std_srvs::srv::Trigger::Request> request,
                                 std::shared_ptr<std_srvs::srv::Trigger::Response> response) {
    if (!robot_) {
        response->success = false;
        response->message = "IMU is not initialized, cannot read IMU.";
        return;
    }
    try {
        response->success = true;
        response->message = "IMU read successfully";
        read_imu();
        publish_imu();
    } catch (const std::exception& e) {
        response->success = false;
        response->message = e.what();
    }
}

void InferenceNode::set_zeros_srv(const std::shared_ptr<std_srvs::srv::Trigger::Request> request,
                                  std::shared_ptr<std_srvs::srv::Trigger::Response> response) {
    if (!robot_->is_init_.load()) {
        response->success = false;
        response->message = "Motors are not initialized, cannot set zeros.";
        return;
    }
    if (is_running_.load()) {
        response->success = false;
        response->message = "Inference is running, cannot set zeros.";
        return;
    }
    try {
        robot_->set_zeros();
        response->success = true;
        response->message = "Zeros set successfully";
    } catch (const std::exception& e) {
        response->success = false;
        response->message = e.what();
    }
}

void InferenceNode::clear_errors_srv(const std::shared_ptr<std_srvs::srv::Trigger::Request> request,
                                     std::shared_ptr<std_srvs::srv::Trigger::Response> response) {
    if (!robot_) {
        response->success = false;
        response->message = "Robot interface is not initialized, cannot clear errors.";
        return;
    }
    try {
        robot_->clear_errors();
        response->success = true;
        response->message = "Errors cleared successfully";
    } catch (const std::exception& e) {
        response->success = false;
        response->message = e.what();
    }
}

void InferenceNode::init_motors_srv(const std::shared_ptr<std_srvs::srv::Trigger::Request> request,
                                    std::shared_ptr<std_srvs::srv::Trigger::Response> response) {
    if (robot_->is_init_.load()) {
        response->success = false;
        response->message = "Motors are already initialized, cannot init motors.";
        return;
    }
    try {
        robot_->init_motors();
        response->success = true;
        response->message = "Motors initialized successfully";
    } catch (const std::exception& e) {
        response->success = false;
        response->message = e.what();
    }
}

void InferenceNode::deinit_motors_srv(const std::shared_ptr<std_srvs::srv::Trigger::Request> request,
                                      std::shared_ptr<std_srvs::srv::Trigger::Response> response) {
    if (!robot_->is_init_.load()) {
        response->success = false;
        response->message = "Motors are not initialized, cannot deinit motors.";
        return;
    }
    try {
        robot_->deinit_motors();
        response->success = true;
        response->message = "Motors deinitialized successfully";
    } catch (const std::exception& e) {
        response->success = false;
        response->message = e.what();
    }
}

void InferenceNode::start_inference_srv(const std::shared_ptr<std_srvs::srv::Trigger::Request> request,
                                        std::shared_ptr<std_srvs::srv::Trigger::Response> response) {
    if (is_running_.load()) {
        response->success = false;
        response->message = "Inference is already running!";
        return;
    }
    is_running_.store(true);
    response->success = true;
    response->message = "Inference started";
}

void InferenceNode::stop_inference_srv(const std::shared_ptr<std_srvs::srv::Trigger::Request> request,
                                       std::shared_ptr<std_srvs::srv::Trigger::Response> response) {
    if (!is_running_.load()) {
        response->success = false;
        response->message = "Inference is already stopped!";
        return;
    }
    is_running_.store(false);
    response->success = true;
    response->message = "Inference stopped";
}

void InferenceNode::publish_joint_states() {
    auto msg = sensor_msgs::msg::JointState();
    msg.header.stamp = this->now();
    for (int i = 0; i < joint_num_; i++) {
        msg.name.push_back("joint_" + std::to_string(i+1));
        msg.position.push_back(joint_pos_[i]);
        msg.velocity.push_back(joint_vel_[i]);
        msg.effort.push_back(joint_torques_[i]);
    }
    joint_state_publisher_->publish(msg);
}

void InferenceNode::publish_action() {
    auto msg = sensor_msgs::msg::JointState();
    msg.header.stamp = this->now();
    for (int i = 0; i < joint_num_; i++) {
        msg.name.push_back("action_" + std::to_string(i+1));
        msg.position.push_back(act_[i]);
    }
    action_publisher_->publish(msg);
}

void InferenceNode::publish_imu() {
    auto msg = sensor_msgs::msg::Imu();
    msg.header.stamp = this->now();
    msg.orientation.w = quat_[0];
    msg.orientation.x = quat_[1];
    msg.orientation.y = quat_[2];
    msg.orientation.z = quat_[3];
    msg.angular_velocity.x = ang_vel_[0];
    msg.angular_velocity.y = ang_vel_[1];
    msg.angular_velocity.z = ang_vel_[2];
    imu_publisher_->publish(msg);
}