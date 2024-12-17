import torch

def select_frames_with_cycling(input_tensors, num_frames=64):
    selected_frames_batch = []
    for input_tensor in input_tensors:
        # 获取张量的总帧数
        total_frames = input_tensor.size(0)

        # 如果总帧数已经大于等于需要的帧数，直接选取前 num_frames 个帧
        if total_frames >= num_frames:
            selected_frames_batch.append(input_tensor[:num_frames])
        else:
            # 计算需要循环的轮数
            num_cycles = num_frames // total_frames
            remainder_frames = num_frames % total_frames

            # 构建新的张量来存储结果
            selected_frames = []

            # 循环补充帧
            for _ in range(num_cycles):
                selected_frames.append(input_tensor)

            # 补充剩余帧
            selected_frames.append(input_tensor[:remainder_frames])

            # 拼接选取的帧
            selected_frames = torch.cat(selected_frames, dim=0)

            selected_frames_batch.append(selected_frames)

    selected_frames_batch = torch.stack(selected_frames_batch, dim=0)  # 将列表转换为张量

    return selected_frames_batch


# 示例用法
if __name__ == '__main__':
    # 假设你有一个批次的视频序列，每个元素是一个形状为 (T, C, H, W) 的张量
    # 这里用随机数初始化一个批次来模拟
    video_batch = [torch.randn(28, 3, 224, 224)]

    # 选取每个视频序列的前64个帧，如果不足64个则循环补充
    selected_frames_batch = select_frames_with_cycling(video_batch, num_frames=64)

    # 打印结果张量的形状
    print(selected_frames_batch.size())
