import torch
import os

def get_end_effector_velocities(file_name_without_ext, frame_number):
    """
    Loads HumanML3D motion data from a .pt file and calculates the L2 norm
    of the velocities for the left foot, right foot, left hand, and right hand,
    and returns root motion parameters for a specific frame.

    Args:
        file_name_without_ext (str): The name of the .pt file without the extension.
        frame_number (int): The specific frame number for which to extract data.

    Returns:
        tuple: A tuple containing:
               - root_angular_velocity_y (float): Root angular velocity around Y-axis.
               - root_linear_velocity_x (float): Root linear velocity along X-axis.
               - root_linear_velocity_z (float): Root linear velocity along Z-axis.
               - left_foot_vel_norm (float): L2 norm of left foot velocity for the frame.
               - right_foot_vel_norm (float): L2 norm of right foot velocity for the frame.
               - left_hand_vel_norm (float): L2 norm of left hand velocity for the frame.
               - right_hand_vel_norm (float): L2 norm of right hand velocity for the frame.
    """
    base_path = "/home/mani/Repos/hcdt/data/humanml3d/"
    file_path = os.path.join(base_path, f"{file_name_without_ext}.pt")

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    data = torch.load(file_path)

    if data.ndim != 2 or data.shape[1] != 263:
        raise ValueError(
            f"Expected data to have shape (num_frames, 263), but got {data.shape}"
        )

    num_frames = data.shape[0]
    if not (0 <= frame_number < num_frames):
        raise ValueError(
            f"frame_number {frame_number} is out of bounds. Must be between 0 and {num_frames - 1}."
        )

    # Get data for the specific frame
    frame_data = data[frame_number, :]

    # Extract root motion parameters (first 3 values)
    root_angular_velocity_y = frame_data[0].item()
    root_linear_velocity_x = frame_data[1].item()
    root_linear_velocity_z = frame_data[2].item()

    # Indices for velocities (vx, vy, vz)
    # Left Foot (Joint 10): 223, 224, 225
    # Right Foot (Joint 11): 226, 227, 228
    # Left Hand (Wrist - Joint 20): 253, 254, 255
    # Right Hand (Wrist - Joint 21): 256, 257, 258

    left_foot_vel_indices = torch.tensor([223, 224, 225])
    right_foot_vel_indices = torch.tensor([226, 227, 228])
    left_hand_vel_indices = torch.tensor([253, 254, 255])
    right_hand_vel_indices = torch.tensor([256, 257, 258])

    left_foot_velocity = frame_data[left_foot_vel_indices]
    right_foot_velocity = frame_data[right_foot_vel_indices]
    left_hand_velocity = frame_data[left_hand_vel_indices]
    right_hand_velocity = frame_data[right_hand_vel_indices]

    # Ensure inputs to linalg.norm are tensors
    left_foot_velocity_tensor = torch.as_tensor(left_foot_velocity, dtype=torch.float32)
    right_foot_velocity_tensor = torch.as_tensor(right_foot_velocity, dtype=torch.float32)
    left_hand_velocity_tensor = torch.as_tensor(left_hand_velocity, dtype=torch.float32)
    right_hand_velocity_tensor = torch.as_tensor(right_hand_velocity, dtype=torch.float32)

    # Calculate L2 norm for each end effector's velocity vector for the specific frame
    left_foot_vel_norm = torch.linalg.norm(left_foot_velocity_tensor).item()
    right_foot_vel_norm = torch.linalg.norm(right_foot_velocity_tensor).item()
    left_hand_vel_norm = torch.linalg.norm(left_hand_velocity_tensor).item()
    right_hand_vel_norm = torch.linalg.norm(right_hand_velocity_tensor).item()
    total_frames=data.shape[0]

    return (
        root_angular_velocity_y,
        root_linear_velocity_x,
        root_linear_velocity_z,
        left_foot_vel_norm,
        right_foot_vel_norm,
        left_hand_vel_norm,
        right_hand_vel_norm,
        total_frames
    )

if __name__ == '__main__':
    # Example usage:
    # Create a dummy file for testing
    data_path = "/home/mani/Repos/hcdt/data/humanml3d/S01A02I01S1.pt"
    test="S01A02I01S1"

    target_frame = 10
    (
        root_ang_vel_y,
        root_lin_vel_x,
        root_lin_vel_z,
        lf_v,
        rf_v,
        lh_v,
        rh_v,
        framecount,
    ) = get_end_effector_velocities(test, target_frame)
    print(f"\nData for frame {target_frame}:")
    print(f"Root Angular Velocity (Y): {root_ang_vel_y:.4f}")
    print(f"Root Linear Velocity (X): {root_lin_vel_x:.4f}")
    print(f"Root Linear Velocity (Z): {root_lin_vel_z:.4f}")
    print(f"Left Foot Velocity Norm: {lf_v:.4f}")
    print(f"Right Foot Velocity Norm: {rf_v:.4f}")
    print(f"Left Hand Velocity Norm: {lh_v:.4f}")
    print(f"Right Hand Velocity Norm: {rh_v:.4f}")
    print(f"Total Frames in sequence: {framecount}")

        



