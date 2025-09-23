#include "inference_node.hpp"

void InferenceNode::subs_cmd_callback(const std::shared_ptr<geometry_msgs::msg::Twist> msg){
    if(!is_joy_control_){
        auto data = std::atomic_load(&write_buffer_); // 原子加载
        data->vx = std::clamp(msg->linear.x, -0.3, 0.3);
        data->vy = std::clamp(msg->linear.y, -0.3, 0.3);
        data->dyaw = std::clamp(msg->angular.z, -0.6, 0.6);
    }
}

void InferenceNode::subs_joy_callback(const std::shared_ptr<sensor_msgs::msg::Joy> msg) {
    if (is_joy_control_){
        auto data = std::atomic_load(&write_buffer_); // 原子加载
        data->vx = std::clamp(msg->axes[3] * 0.3, -0.3, 0.3);
        data->vy = std::clamp(msg->axes[2] * 0.3, -0.3, 0.3);
            if (msg->buttons[6] == 1) {
            data->dyaw = std::clamp(msg->buttons[6] * 0.6, 0.0, 0.6);
            } else if (msg->buttons[7] == 1) {
            data->dyaw = std::clamp(-msg->buttons[7] * 0.6, -0.6, 0.0);
            } else {
            data->dyaw = 0.0;
        }
    }
    if ((msg->buttons[0] == 1 && msg->buttons[0] != last_button0_) || (msg->buttons[1] == 1 && msg->buttons[1] != last_button1_) || (msg->buttons[3] == 1 && msg->buttons[3] != last_button3_)) {
        is_running_.store(false);
        hist_obs_.clear();
        last_output_ = std::vector<float>(23, 0.0);
        last_act_ = std::vector<float>(23, 0.0);
        act_ = std::make_shared<std::vector<float>>(23, 0.0);
        write_buffer_ = std::make_shared<SensorData>();
        write_buffer_->imu_obs[0] = 1.0;
        read_buffer_ = std::make_shared<SensorData>();
        read_buffer_->imu_obs[0] = 1.0;
        is_first_frame_ = true;
        RCLCPP_INFO(this->get_logger(), "Inference paused");
        if (msg->buttons[3] == 1){
            is_joy_control_.store(!is_joy_control_);
            RCLCPP_INFO(this->get_logger(), "Controlled by %s", is_joy_control_.load() ? "joy" : "/cmd_vel");
        }
    }
    if (msg->buttons[2] == 1 && msg->buttons[2] != last_button2_) {
        is_running_.store(!is_running_.load());
        RCLCPP_INFO(this->get_logger(), "Inference %s", is_running_.load() ? "started" : "paused");
    }
    last_button0_ = msg->buttons[0];
    last_button1_ = msg->buttons[1];
    last_button2_ = msg->buttons[2];
    last_button3_ = msg->buttons[3];
}

void InferenceNode::subs_left_leg_callback(const std::shared_ptr<sensor_msgs::msg::JointState> msg) {
    auto data = std::atomic_load(&write_buffer_); // 原子加载
    for (int i = 0; i < 6; i++) {
        if (msg->position[i] > joint_limits_upper_[i] || msg->position[i] < joint_limits_lower_[i]){
            is_running_.store(false);
            RCLCPP_WARN(this->get_logger(), "Left leg joint %d out of limits, inference paused!", i+1);
            return;
        }
        data->left_leg_obs[i] = msg->position[i];
        data->left_leg_obs[6 + i] = msg->velocity[i];
    }
}

void InferenceNode::subs_right_leg_callback(const std::shared_ptr<sensor_msgs::msg::JointState> msg) {
    auto data = std::atomic_load(&write_buffer_); // 原子加载
    for (int i = 0; i < 7; i++) {
        if (msg->position[i] > joint_limits_upper_[6+i] || msg->position[i] < joint_limits_lower_[6+i]){
            is_running_.store(false);
            RCLCPP_WARN(this->get_logger(), "Right leg joint %d out of limits, inference paused!", i+1);
            return;
        }
        data->right_leg_obs[i] = msg->position[i];
        data->right_leg_obs[7 + i] = msg->velocity[i];
    }
}

void InferenceNode::subs_left_arm_callback(const std::shared_ptr<sensor_msgs::msg::JointState> msg) {
    auto data = std::atomic_load(&write_buffer_); // 原子加载
    for (int i = 0; i < 5; i++) {
        if (msg->position[i] > joint_limits_upper_[13+i] || msg->position[i] < joint_limits_lower_[13+i]){
            is_running_.store(false);
            RCLCPP_WARN(this->get_logger(), "Left arm joint %d out of limits, inference paused!", i+1);
            return;
        }
        data->left_arm_obs[i] = msg->position[i];
        data->left_arm_obs[5 + i] = msg->velocity[i];
    }
}

void InferenceNode::subs_right_arm_callback(const std::shared_ptr<sensor_msgs::msg::JointState> msg) {
    auto data = std::atomic_load(&write_buffer_); // 原子加载
    for (int i = 0; i < 5; i++) {
        if (msg->position[i] > joint_limits_upper_[18+i] || msg->position[i] < joint_limits_lower_[18+i]){
            is_running_.store(false);
            RCLCPP_WARN(this->get_logger(), "Right arm joint %d out of limits, inference paused!", i+1);
            return;
        }
        data->right_arm_obs[i] = msg->position[i];
        data->right_arm_obs[5 + i] = msg->velocity[i];
    }
}

void InferenceNode::subs_IMU_callback(const std::shared_ptr<sensor_msgs::msg::Imu> msg) {
    auto data = std::atomic_load(&write_buffer_); // 原子加载
    data->imu_obs[0] = msg->orientation.w;
    data->imu_obs[1] = msg->orientation.x;
    data->imu_obs[2] = msg->orientation.y;
    data->imu_obs[3] = msg->orientation.z;
    data->imu_obs[4] = gyro_alpha_ * msg->angular_velocity.x + (1 - gyro_alpha_) * data->imu_obs[4];
    data->imu_obs[5] = gyro_alpha_ * msg->angular_velocity.y + (1 - gyro_alpha_) * data->imu_obs[5];
    data->imu_obs[6] = gyro_alpha_ * msg->angular_velocity.z + (1 - gyro_alpha_) * data->imu_obs[6];
}

void InferenceNode::publish_joint_states() {
    if(!is_running_.load()){
        return;
    }
    auto act = std::atomic_load(&act_); // 原子加载
    for (size_t i = 0; i < act->size(); i++) {
        (*act)[i] = act_alpha_ * (*act)[i] + (1 - act_alpha_) * last_act_[i];
    }
    last_act_ = *act;
    auto left_leg_message = sensor_msgs::msg::JointState();
    left_leg_message.header.stamp = this->now();
    left_leg_message.name = {"joint1", "joint2", "joint3", "joint4", "joint5", "joint6"};
    left_leg_message.position = {(*act)[0], (*act)[1], (*act)[2], (*act)[3], (*act)[4], (*act)[5]};
    left_leg_message.velocity = {0, 0, 0, 0, 0, 0};
    left_leg_message.effort = {0, 0, 0, 0, 0, 0};
    left_leg_publisher_->publish(left_leg_message);

    auto right_leg_message = sensor_msgs::msg::JointState();
    right_leg_message.header.stamp = this->now();
    right_leg_message.name = {"joint1", "joint2", "joint3", "joint4", "joint5", "joint6", "joint7"};
    right_leg_message.position = {(*act)[6], (*act)[7], (*act)[8], (*act)[9], (*act)[10], (*act)[11], (*act)[12]};
    right_leg_message.velocity = {0, 0, 0, 0, 0, 0, 0};
    right_leg_message.effort = {0, 0, 0, 0, 0, 0, 0};
    right_leg_publisher_->publish(right_leg_message);

    auto left_arm_message = sensor_msgs::msg::JointState();
    left_arm_message.header.stamp = this->now();
    left_arm_message.name = {"joint1", "joint2", "joint3", "joint4", "joint5"};
    left_arm_message.position = {(*act)[13], (*act)[14], (*act)[15], (*act)[16], (*act)[17]};
    left_arm_message.velocity = {0, 0, 0, 0, 0};
    left_arm_message.effort = {0, 0, 0, 0, 0};
    left_arm_publisher_->publish(left_arm_message);

    auto right_arm_message = sensor_msgs::msg::JointState();
    right_arm_message.header.stamp = this->now();
    right_arm_message.name = {"joint1", "joint2", "joint3", "joint4", "joint5"};
    right_arm_message.position = {(*act)[18], (*act)[19], (*act)[20], (*act)[21], (*act)[22]};
    right_arm_message.velocity = {0, 0, 0, 0, 0};
    right_arm_message.effort = {0, 0, 0, 0, 0};
    right_arm_publisher_->publish(right_arm_message);
}

void InferenceNode::get_gravity_b(const SensorData& data) {
    float w, x, y, z;
    w = data.imu_obs[0];
    x = data.imu_obs[1];
    y = data.imu_obs[2];
    z = data.imu_obs[3];

    Eigen::Quaternionf q_b2w(w, x, y, z);
    Eigen::Vector3f gravity_w(0.0f, 0.0f, -1.0f);
    Eigen::Quaternionf q_w2b = q_b2w.inverse();
    Eigen::Vector3f gravity_b = q_w2b * gravity_w;
    if (gravity_b.z() > gravity_z_upper_){
        is_running_.store(false);
        RCLCPP_WARN(this->get_logger(), "Robot fell down! Inference paused.");
        return;
    }

    obs_[3] = gravity_b.x() * obs_scales_gravity_b_;
    obs_[4] = gravity_b.y() * obs_scales_gravity_b_;
    obs_[5] = gravity_b.z() * obs_scales_gravity_b_;

    // RCLCPP_INFO(this->get_logger(), "gravity_b: %f %f %f", obs_[44], obs_[45], obs_[46]);
}

void InferenceNode::inference() {
    pthread_setname_np(pthread_self(), "inference");
    struct sched_param sp{}; sp.sched_priority = 70;
    pthread_setschedparam(pthread_self(), SCHED_FIFO, &sp);
    auto period = std::chrono::microseconds(static_cast<long long>(dt_ * 1000 * 1000 * decimation_));

    while(rclcpp::ok()){
        auto loop_start = std::chrono::steady_clock::now();
        if(!is_running_.load()){
            std::this_thread::sleep_for(period);
            continue;
        }

        write_buffer_ = std::atomic_exchange(&read_buffer_, write_buffer_);
        auto data = std::atomic_load(&read_buffer_); 

        for (int i = 0; i < 3; i++) {
            obs_[i] = data->imu_obs[4 + i] * obs_scales_ang_vel_;
        }
        get_gravity_b(*data);
        obs_[6] = data->vx * obs_scales_lin_vel_;
        obs_[7] = data->vy * obs_scales_lin_vel_;
        obs_[8] = data->dyaw * obs_scales_ang_vel_;
        // RCLCPP_INFO(this->get_logger(), "obs_[4]: %f", obs_[4]);

        std::vector<float> joint_obs;
        joint_obs.resize(46);
        for (int i = 0; i < 6; i++) {
            joint_obs[i] = data->left_leg_obs[i] * obs_scales_dof_pos_;
            joint_obs[23 + i] = data->left_leg_obs[6 + i] * obs_scales_dof_vel_;
        }
        for (int i = 0; i < 7; i++) {
            joint_obs[6 + i] = data->right_leg_obs[i] * obs_scales_dof_pos_;
            joint_obs[29 + i] = data->right_leg_obs[7 + i] * obs_scales_dof_vel_;
        }
        for (int i = 0; i < 5; i++) {
            joint_obs[13 + i] = data->left_arm_obs[i] * obs_scales_dof_pos_;
            joint_obs[36 + i] = data->left_arm_obs[5 + i] * obs_scales_dof_vel_;
        }
        for (int i = 0; i < 5; i++) {
            joint_obs[18 + i] = data->right_arm_obs[i] * obs_scales_dof_pos_;
            joint_obs[41 + i] = data->right_arm_obs[5 + i] * obs_scales_dof_vel_;
        }
        for (int i = 0; i < 23; i++) {
            obs_[9 + i] = joint_obs[usd2urdf_[i]];
            obs_[32 + i] = joint_obs[23 + usd2urdf_[i]];
        }

        for (int i = 0; i < 23; i++) {
            obs_[55 + i] = last_output_[i];
        }
        std::transform(obs_.begin(), obs_.end(), obs_.begin(), [this](float val) {
            return std::clamp(val, -clip_observations_, clip_observations_);
        });
        if (is_first_frame_) {
            for (int i = 0; i < frame_stack_; i++) {
                hist_obs_.push_back(obs_);
            }
            is_first_frame_ = false;
        } else {
            hist_obs_.pop_front();
            hist_obs_.push_back(obs_);
        }

        std::vector<float> input(78 * frame_stack_);
        for (int i = 0; i < frame_stack_; i++) {
            std::copy(hist_obs_[i].begin(), hist_obs_[i].end(), input.begin() + i * 78);
        }
        auto memory_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
        std::vector<const char *> input_names_raw(num_inputs_);
        for (size_t i = 0; i < num_inputs_; i++) {
            input_names_raw[i] = input_names_[i].c_str();
        }
        std::vector<const char *> output_names_raw(num_outputs_);
        for (size_t i = 0; i < num_outputs_; i++) {
            output_names_raw[i] = output_names_[i].c_str();
        }
        Ort::Value input_tensor =
            Ort::Value::CreateTensor<float>(memory_info, const_cast<float *>(input.data()), input.size(),
                                            input_shape_.data(), input_shape_.size());

        auto output_tensors = session_->Run(Ort::RunOptions{nullptr}, input_names_raw.data(), &input_tensor,
                                            1, output_names_raw.data(), output_names_raw.size());

        std::vector<float> output;
        for (auto &tensor : output_tensors) {
            float *data = tensor.GetTensorMutableData<float>();
            size_t count = tensor.GetTensorTypeAndShapeInfo().GetElementCount();
            output.insert(output.end(), data, data + count);
        }

        auto new_act = std::make_shared<std::vector<float>>(output.size());
        for (int i = 0; i < output.size(); i++) {
            output[i] = std::clamp(output[i], -clip_actions_, clip_actions_);
            (*new_act)[usd2urdf_[i]] = output[i];
            (*new_act)[usd2urdf_[i]] = (*new_act)[usd2urdf_[i]] * action_scale_;
        }
        std::atomic_store(&act_, new_act);
        last_output_ = output;


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

int main(int argc, char **argv) {
    rclcpp::init(argc, argv);
    auto node = std::make_shared<InferenceNode>();
    rclcpp::executors::MultiThreadedExecutor executor(rclcpp::ExecutorOptions(), 4);
    executor.add_node(node);
    RCLCPP_INFO(node->get_logger(), "Press 'B' to start/pause inference");
    executor.spin();
    rclcpp::shutdown();
    return 0;
}