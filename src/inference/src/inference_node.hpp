#include <onnxruntime_cxx_api.h>
#include <string.h>

#include <atomic>
#include <memory>
#include <Eigen/Geometry>
#include <algorithm>
#include <cmath>
#include <chrono>
#include <filesystem>
#include <iostream>
#include <queue>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/imu.hpp>
#include <sensor_msgs/msg/joint_state.hpp>
#include <sensor_msgs/msg/joy.hpp>
#include <vector>

class InferenceNode : public rclcpp::Node {
   public:
    InferenceNode() : Node("inference_node") {
        write_buffer_ = std::make_shared<SensorData>();
        write_buffer_->imu_obs[0] = 1.0;
        read_buffer_ = std::make_shared<SensorData>();
        read_buffer_->imu_obs[0] = 1.0;
        obs_.resize(78);
        act_ = std::make_shared<std::vector<float>>(23);
        last_act_.resize(23);
        usd2urdf_.resize(23);
        last_output_.resize(23);
        gravity_z_upper_ = -0.7;
        joint_limits_lower_ = std::vector<float>{-1.0,-0.2,-1.0+0.24,-0.2-0.48,-0.6+0.24,-0.5,-0.2,-1.0,-1.0+0.24,-0.2-0.48,-0.6+0.24,-0.5,-3.14,-1.57-0.1,-0.25-0.07,-1.57,-0.6-0.1,-1.57,-1.57-0.1,-1.0+0.07,-1.57,-0.6-1.0,-1.57};
        joint_limits_upper_ = std::vector<float>{ 0.2, 1.0, 1.0+0.24, 2.5-0.48, 0.6+0.24, 0.5, 1.0, 0.2, 1.0+0.24, 2.5-0.48, 0.6+0.24, 0.5, 3.14, 1.57-0.1,  1.0-0.07, 1.57,1.57-0.1, 1.57, 1.57-0.1,0.25+0.07, 1.57,1.57-1.0, 1.57};
        is_first_frame_ = true;

        this->declare_parameter<std::string>("model_name", "1.onnx");
        this->declare_parameter<float>("act_alpha", 0.9);
        this->declare_parameter<float>("gyro_alpha", 0.9);
        this->declare_parameter<float>("angle_alpha", 0.9);
        this->declare_parameter<int>("intra_threads", -1);
        this->declare_parameter<int>("frame_stack", 15);
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
        this->declare_parameter<std::vector<long int>>(
            "usd2urdf", std::vector<long int>{0, 6,  12, 1, 7,  13, 18, 2, 8,  14, 19, 3,
                                              9, 15, 20, 4, 10, 16, 21, 5, 11, 17, 22});

        this->get_parameter("model_name", model_name_);
        this->get_parameter("act_alpha", act_alpha_);
        this->get_parameter("gyro_alpha", gyro_alpha_);
        this->get_parameter("angle_alpha", angle_alpha_);
        this->get_parameter("intra_threads", intra_threads_);
        this->get_parameter("frame_stack", frame_stack_);
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
        this->get_parameter("usd2urdf", usd2urdf_);

        model_path_ = std::string(ROOT_DIR) + "models/" + model_name_;
        RCLCPP_INFO(this->get_logger(), "model_path: %s", model_path_.c_str());
        RCLCPP_INFO(this->get_logger(), "act_alpha: %f", act_alpha_);
        RCLCPP_INFO(this->get_logger(), "gyro_alpha: %f", gyro_alpha_);
        RCLCPP_INFO(this->get_logger(), "angle_alpha: %f", angle_alpha_);
        RCLCPP_INFO(this->get_logger(), "intra_threads: %d", intra_threads_);
        RCLCPP_INFO(this->get_logger(), "frame_stack: %d", frame_stack_);
        RCLCPP_INFO(this->get_logger(), "decimation: %d", decimation_);
        RCLCPP_INFO(this->get_logger(), "dt: %f", dt_);
        RCLCPP_INFO(this->get_logger(), "obs_scales_lin_vel: %f", obs_scales_lin_vel_);
        RCLCPP_INFO(this->get_logger(), "obs_scales_ang_vel: %f", obs_scales_ang_vel_);
        RCLCPP_INFO(this->get_logger(), "obs_scales_dof_pos: %f", obs_scales_dof_pos_);
        RCLCPP_INFO(this->get_logger(), "obs_scales_dof_vel: %f", obs_scales_dof_vel_);
        RCLCPP_INFO(this->get_logger(), "obs_scales_gravity_b: %f", obs_scales_gravity_b_);
        RCLCPP_INFO(this->get_logger(), "action_scale: %f", action_scale_);
        RCLCPP_INFO(this->get_logger(), "clip_actions: %f", clip_actions_);
        RCLCPP_INFO(this->get_logger(),
                    "usd2urdf: %ld, %ld, %ld, %ld, %ld, %ld, %ld, %ld, %ld, %ld, %ld, %ld, %ld, %ld, %ld, %ld, %ld, %ld, %ld, "
                    "%ld, %ld, %ld, %ld",
                    usd2urdf_[0], usd2urdf_[1], usd2urdf_[2], usd2urdf_[3], usd2urdf_[4], usd2urdf_[5],
                    usd2urdf_[6], usd2urdf_[7], usd2urdf_[8], usd2urdf_[9], usd2urdf_[10], usd2urdf_[11],
                    usd2urdf_[12], usd2urdf_[13], usd2urdf_[14], usd2urdf_[15], usd2urdf_[16], usd2urdf_[17],
                    usd2urdf_[18], usd2urdf_[19], usd2urdf_[20], usd2urdf_[21], usd2urdf_[22]);
        
        Ort::ThreadingOptions thread_opts;
        thread_opts.SetGlobalSpinControl(false);
        if (intra_threads_ > 0) {
            thread_opts.SetGlobalIntraOpNumThreads(intra_threads_);
        }
        env_ = std::make_unique<Ort::Env>(thread_opts, ORT_LOGGING_LEVEL_WARNING, "ONNXRuntimeInference");
        Ort::SessionOptions session_options;
        session_options.DisablePerSessionThreads();
        session_options.EnableCpuMemArena();
        session_options.EnableMemPattern();
        // session_options.EnableProfiling("onnxruntime_profile.json");
        session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
        session_ = std::make_unique<Ort::Session>(*env_, model_path_.c_str(), session_options);
        num_inputs_ = session_->GetInputCount();
        input_names_.resize(num_inputs_);
        for (size_t i = 0; i < num_inputs_; i++) {
            Ort::AllocatedStringPtr input_name = session_->GetInputNameAllocated(i, allocator_);
            input_names_[i] = input_name.get();
            auto type_info = session_->GetInputTypeInfo(i);
            input_shape_ = type_info.GetTensorTypeAndShapeInfo().GetShape();
        }
        num_outputs_ = session_->GetOutputCount();
        output_names_.resize(num_outputs_);
        for (size_t i = 0; i < num_outputs_; i++) {
            Ort::AllocatedStringPtr output_name = session_->GetOutputNameAllocated(i, allocator_);
            output_names_[i] = output_name.get();
        }

        auto sensor_data_qos = rclcpp::QoS(rclcpp::KeepLast(1)).best_effort().durability_volatile();
        auto control_command_qos = rclcpp::QoS(rclcpp::KeepLast(1)).reliable().durability_volatile();
        left_leg_subscription_ = this->create_subscription<sensor_msgs::msg::JointState>(
            "/joint_states_left_leg", sensor_data_qos,
            std::bind(&InferenceNode::subs_left_leg_callback, this, std::placeholders::_1));
        right_leg_subscription_ = this->create_subscription<sensor_msgs::msg::JointState>(
            "/joint_states_right_leg", sensor_data_qos,
            std::bind(&InferenceNode::subs_right_leg_callback, this, std::placeholders::_1));
        left_arm_subscription_ = this->create_subscription<sensor_msgs::msg::JointState>(
            "/joint_states_left_arm", sensor_data_qos,
            std::bind(&InferenceNode::subs_left_arm_callback, this, std::placeholders::_1));
        right_arm_subscription_ = this->create_subscription<sensor_msgs::msg::JointState>(
            "/joint_states_right_arm", sensor_data_qos,
            std::bind(&InferenceNode::subs_right_arm_callback, this, std::placeholders::_1));
        IMU_subscription_ = this->create_subscription<sensor_msgs::msg::Imu>(
            "/IMU_data", sensor_data_qos, std::bind(&InferenceNode::subs_IMU_callback, this, std::placeholders::_1));
        joy_subscription_ = this->create_subscription<sensor_msgs::msg::Joy>(
            "/joy", sensor_data_qos, std::bind(&InferenceNode::subs_joy_callback, this, std::placeholders::_1));
        left_leg_publisher_ =
            this->create_publisher<sensor_msgs::msg::JointState>("/joint_command_left_leg", control_command_qos);
        right_leg_publisher_ =
            this->create_publisher<sensor_msgs::msg::JointState>("/joint_command_right_leg", control_command_qos);
        left_arm_publisher_ =
            this->create_publisher<sensor_msgs::msg::JointState>("/joint_command_left_arm", control_command_qos);
        right_arm_publisher_ =
            this->create_publisher<sensor_msgs::msg::JointState>("/joint_command_right_arm", control_command_qos);
        inference_thread_ = std::thread(&InferenceNode::inference, this);
        timer_pub_ = this->create_wall_timer(std::chrono::milliseconds((int)(dt_ * 1000)),
                                             std::bind(&InferenceNode::publish_joint_states, this));
    }
    ~InferenceNode() {
        if (inference_thread_.joinable()) {
            inference_thread_.join();
        }
    }

   private:
   struct SensorData {
        float vx = 0.0, vy = 0.0, dyaw = 0.0;
        std::vector<float> left_leg_obs = std::vector<float>(12, 0.0);
        std::vector<float> right_leg_obs = std::vector<float>(14, 0.0);
        std::vector<float> left_arm_obs = std::vector<float>(10, 0.0);
        std::vector<float> right_arm_obs = std::vector<float>(10, 0.0);
        std::vector<float> imu_obs = std::vector<float>(7, 0.0);
    };
    std::shared_ptr<SensorData> write_buffer_, read_buffer_;
    std::atomic<bool> is_running_{false};
    std::string model_name_, model_path_;
    int frame_stack_;
    int decimation_;
    std::unique_ptr<Ort::Env> env_;
    std::unique_ptr<Ort::Session> session_;
    std::vector<std::string> input_names_, output_names_;
    size_t num_inputs_, num_outputs_;
    std::vector<int64_t> input_shape_;
    int intra_threads_;
    Ort::AllocatorWithDefaultOptions allocator_;
    rclcpp::Publisher<sensor_msgs::msg::JointState>::SharedPtr left_leg_publisher_, right_leg_publisher_,
        left_arm_publisher_, right_arm_publisher_;
    rclcpp::Subscription<sensor_msgs::msg::JointState>::SharedPtr left_leg_subscription_,
        right_leg_subscription_, left_arm_subscription_, right_arm_subscription_;
    rclcpp::Subscription<sensor_msgs::msg::Imu>::SharedPtr  IMU_subscription_;
    rclcpp::Subscription<sensor_msgs::msg::Joy>::SharedPtr joy_subscription_;
    rclcpp::TimerBase::SharedPtr timer_pub_;
    std::thread inference_thread_;
    std::vector<float> obs_, last_act_, last_output_;
    std::shared_ptr<std::vector<float>> act_;
    float act_alpha_, gyro_alpha_, angle_alpha_;
    std::deque<std::vector<float>> hist_obs_;
    float dt_;
    float obs_scales_lin_vel_, obs_scales_ang_vel_, obs_scales_dof_pos_, obs_scales_dof_vel_,
        obs_scales_gravity_b_, clip_observations_;
    float action_scale_, clip_actions_;
    std::vector<long int> usd2urdf_;
    bool is_first_frame_;
    std::vector<float> joint_limits_lower_, joint_limits_upper_;
    float gravity_z_upper_;
    int last_button0_ = 0, last_button1_ = 0, last_button2_ = 0, last_button3_ = 0;

    void subs_joy_callback(const std::shared_ptr<sensor_msgs::msg::Joy> msg);
    void subs_left_leg_callback(const std::shared_ptr<sensor_msgs::msg::JointState> msg);
    void subs_right_leg_callback(const std::shared_ptr<sensor_msgs::msg::JointState> msg);
    void subs_left_arm_callback(const std::shared_ptr<sensor_msgs::msg::JointState> msg);
    void subs_right_arm_callback(const std::shared_ptr<sensor_msgs::msg::JointState> msg);
    void subs_IMU_callback(const std::shared_ptr<sensor_msgs::msg::Imu> msg);
    void publish_joint_states();
    void get_gravity_b(const SensorData& data);
    void inference();
};