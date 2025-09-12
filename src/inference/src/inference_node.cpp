#include "inference_node.hpp"

void InferenceNode::subs_joy_callback(const std::shared_ptr<sensor_msgs::msg::Joy> msg) {
    auto data = std::atomic_load(&write_buffer_); // 原子加载
    data->vx = msg->axes[3] * 0.3;
    data->vy = msg->axes[2] * 0.3;
    if (msg->buttons[6] == 1) {
        data->dyaw = msg->buttons[6] * 0.8;
    } else if (msg->buttons[7] == 1) {
        data->dyaw = -msg->buttons[7] * 0.8;
    } else {
        data->dyaw = 0.0;
    }
    if (msg->buttons[0] == 1) {
        is_running_ = false;
        RCLCPP_INFO(this->get_logger(), "Inference paused");
    }
    if (msg->buttons[1] == 1) {
        is_running_ = false;
        RCLCPP_INFO(this->get_logger(), "Inference paused");
    }
    if (msg->buttons[2] == 1) {
        is_running_ = true;
        RCLCPP_INFO(this->get_logger(), "Inference started");
    }
    if (msg->buttons[3] == 1) {
        is_running_ = false;
        RCLCPP_INFO(this->get_logger(), "Inference paused");
    }
}

void InferenceNode::subs_left_leg_callback(const std::shared_ptr<sensor_msgs::msg::JointState> msg) {
    auto data = std::atomic_load(&write_buffer_); // 原子加载
    for (int i = 0; i < 6; i++) {
        if (msg->position[i] > joint_limits_upper_[i] || msg->position[i] < joint_limits_lower_[i]){
            is_running_ = false;
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
            is_running_ = false;
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
            is_running_ = false;
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
            is_running_ = false;
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
    auto left_leg_message = sensor_msgs::msg::JointState();
    left_leg_message.header.stamp = this->now();
    left_leg_message.name = {"joint1", "joint2", "joint3", "joint4", "joint5", "joint6"};
    left_leg_message.position = {act_[0], act_[1], act_[2], act_[3], act_[4], act_[5]};
    left_leg_message.velocity = {0, 0, 0, 0, 0, 0};
    left_leg_message.effort = {0, 0, 0, 0, 0, 0};
    left_leg_publisher_->publish(left_leg_message);

    auto right_leg_message = sensor_msgs::msg::JointState();
    right_leg_message.header.stamp = this->now();
    right_leg_message.name = {"joint1", "joint2", "joint3", "joint4", "joint5", "joint6", "joint7"};
    right_leg_message.position = {act_[6], act_[7], act_[8], act_[9], act_[10], act_[11], act_[12]};
    right_leg_message.velocity = {0, 0, 0, 0, 0, 0, 0};
    right_leg_message.effort = {0, 0, 0, 0, 0, 0, 0};
    right_leg_publisher_->publish(right_leg_message);

    auto left_arm_message = sensor_msgs::msg::JointState();
    left_arm_message.header.stamp = this->now();
    left_arm_message.name = {"joint1", "joint2", "joint3", "joint4", "joint5"};
    left_arm_message.position = {act_[13], act_[14], act_[15], act_[16], act_[17]};
    left_arm_message.velocity = {0, 0, 0, 0, 0};
    left_arm_message.effort = {0, 0, 0, 0, 0};
    left_arm_publisher_->publish(left_arm_message);

    auto right_arm_message = sensor_msgs::msg::JointState();
    right_arm_message.header.stamp = this->now();
    right_arm_message.name = {"joint1", "joint2", "joint3", "joint4", "joint5"};
    right_arm_message.position = {act_[18], act_[19], act_[20], act_[21], act_[22]};
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
        is_running_ = false;
        RCLCPP_WARN(this->get_logger(), "Robot fell down! Inference paused.");
        return;
    }

    obs_[3] = gravity_b.x() * obs_scales_gravity_b_;
    obs_[4] = gravity_b.y() * obs_scales_gravity_b_;
    obs_[5] = gravity_b.z() * obs_scales_gravity_b_;

    // RCLCPP_INFO(this->get_logger(), "gravity_b: %f %f %f", obs_[44], obs_[45], obs_[46]);
}

void InferenceNode::inference() {
    if(!is_running_){
        last_output_ = std::vector<float>(23, 0.0);
        last_act_ = std::vector<float>(23, 0.0);
        is_first_frame_ = true;
        return;
    }
    if (step_ % decimation_ == 0) {
        {
            auto new_write_buffer = std::make_shared<SensorData>();
            auto read_buffer = std::atomic_load(&write_buffer_);
            *new_write_buffer = *read_buffer;
            std::atomic_store(&write_buffer_, new_write_buffer);
            for (int i = 0; i < 3; i++) {
                obs_[i] = read_buffer->imu_obs[4 + i] * obs_scales_ang_vel_;
            }
            get_gravity_b(*read_buffer);
            obs_[6] = read_buffer->vx * obs_scales_lin_vel_;
            obs_[7] = read_buffer->vy * obs_scales_lin_vel_;
            obs_[8] = read_buffer->dyaw * obs_scales_ang_vel_;
            // RCLCPP_INFO(this->get_logger(), "obs_[4]: %f", obs_[4]);

            std::vector<float> joint_obs;
            joint_obs.resize(46);
            for (int i = 0; i < 6; i++) {
                joint_obs[i] = read_buffer->left_leg_obs[i] * obs_scales_dof_pos_;
                joint_obs[23 + i] = read_buffer->left_leg_obs[6 + i] * obs_scales_dof_vel_;
            }
            for (int i = 0; i < 7; i++) {
                joint_obs[6 + i] = read_buffer->right_leg_obs[i] * obs_scales_dof_pos_;
                joint_obs[29 + i] = read_buffer->right_leg_obs[7 + i] * obs_scales_dof_vel_;
            }
            for (int i = 0; i < 5; i++) {
                joint_obs[13 + i] = read_buffer->left_arm_obs[i] * obs_scales_dof_pos_;
                joint_obs[36 + i] = read_buffer->left_arm_obs[5 + i] * obs_scales_dof_vel_;
            }
            for (int i = 0; i < 5; i++) {
                joint_obs[18 + i] = read_buffer->right_arm_obs[i] * obs_scales_dof_pos_;
                joint_obs[41 + i] = read_buffer->right_arm_obs[5 + i] * obs_scales_dof_vel_;
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
        }
        // auto start_time = std::chrono::high_resolution_clock::now();
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
        // auto end_time = std::chrono::high_resolution_clock::now();
        // auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end_time - start_time);
        // RCLCPP_INFO(this->get_logger(), "Inference took %lld ns.", duration.count());
        act_.resize(output.size());
        for (int i = 0; i < output.size(); i++) {
            output[i] = std::clamp(output[i], -clip_actions_, clip_actions_);
            act_[usd2urdf_[i]] = output[i];
            act_[usd2urdf_[i]] = act_[usd2urdf_[i]] * action_scale_;
        }
        last_output_ = output;
    }
    for (size_t i = 0; i < act_.size(); i++) {
        act_[i] = act_alpha_ * act_[i] + (1 - act_alpha_) * last_act_[i];
    }
    publish_joint_states();
    last_act_ = act_;
    step_ += 1;
}

int main(int argc, char **argv) {
    rclcpp::init(argc, argv);
    auto node = std::make_shared<InferenceNode>();
    RCLCPP_INFO(node->get_logger(), "Press 'B' to start inference");
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}