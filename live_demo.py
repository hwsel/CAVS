import fcntl
import json
import os
import queue
import subprocess
# import tempfile
import time
from argparse import ArgumentParser
from multiprocessing import Process, Queue
from pathlib import Path
from typing import Iterator

import numpy
import torch
import torchvision
from torch._C._profiler import ProfilerActivity
from torch.autograd.profiler import record_function
from torch.profiler import profile

VIDEOS_NAME = ['Beauty', 'Bosphorus', 'CityAlley', 'FlowerFocus', 'FlowerKids', 'FlowerPan', 'HoneyBee', 'Jockey',
               'Lips', 'RaceNight', 'ReadySteadyGo', 'RiverBank', 'ShakeNDry', 'SunBath', 'Twilight', 'YachtRide', 'Wood']
VIDEOS = {
    'Beauty': (
        f'Beauty_3840x2160.yuv',
        (3840, 2160),
        120,
        ('rawvideo', 'yuv420p')
    ),
    'Bosphorus': (
        f'Bosphorus_3840x2160.yuv',
        (3840, 2160),
        120,
        ('rawvideo', 'yuv420p')
    ),
    'CityAlley': (
        f'CityAlley_3840x2160_50fps_8bit.yuv',
        (3840, 2160),
        50,
        ('rawvideo', 'yuv420p')
    ),
    'FlowerFocus': (
        f'FlowerFocus_3840x2160_50fps_8bit.yuv',
        (3840, 2160),
        50,
        ('rawvideo', 'yuv420p')
    ),
    'FlowerKids': (
        f'FlowerKids_3840x2160_50fps_8bit.yuv',
        (3840, 2160),
        50,
        ('rawvideo', 'yuv420p')
    ),
    'FlowerPan': (
        f'FlowerPan_3840x2160_50fps_8bit.yuv',
        (3840, 2160),
        50,
        ('rawvideo', 'yuv420p')
    ),
    'HoneyBee': (
        f'HoneyBee_3840x2160.yuv',
        (3840, 2160),
        120,
        ('rawvideo', 'yuv420p')
    ),
    'Jockey': (
        f'Jockey_3840x2160.yuv',
        (3840, 2160),
        120,
        ('rawvideo', 'yuv420p')
    ),
    'Lips': (
        f'Lips_3840x2160_120fps_8bit.yuv',
        (3840, 2160),
        120,
        ('rawvideo', 'yuv420p')
    ),
    'RaceNight': (
        f'RaceNight_3840x2160_50fps_8bit.yuv',
        (3840, 2160),
        50,
        ('rawvideo', 'yuv420p')
    ),
    'ReadySteadyGo': (
        f'ReadySteadyGo_3840x2160.yuv',
        (3840, 2160),
        120,
        ('rawvideo', 'yuv420p')
    ),
    'RiverBank': (
        f'RiverBank_3840x2160_50fps_8bit.yuv',
        (3840, 2160),
        50,
        ('rawvideo', 'yuv420p')
    ),
    'ShakeNDry': (
        f'ShakeNDry_3840x2160.yuv',
        (3840, 2160),
        120,
        ('rawvideo', 'yuv420p')
    ),
    'SunBath': (
        f'SunBath_3840x2160_50fps_8bit.yuv',
        (3840, 2160),
        50,
        ('rawvideo', 'yuv420p')
    ),
    'Twilight': (
        f'Twilight_3840x2160_50fps_8bit.yuv',
        (3840, 2160),
        50,
        ('rawvideo', 'yuv420p')
    ),
    'YachtRide': (
        f'YachtRide_3840x2160.yuv',
        (3840, 2160),
        120,
        ('rawvideo', 'yuv420p')
    ),
    'Wood': (
        f'Wood.yuv',
        (3840, 2160),
        120,
        ('rawvideo', 'yuv420p')
    ),
}  # type: dict[str, tuple[str, tuple[int, int], int, tuple[str, str]]]

ESPCN = {
    2: f'epoch_2_100.pt',
    3: f'epoch_3_100.pt',
    4: f'epoch_4_100.pt',
    8: f'epoch_8_100.pt',
}  # type: dict[int, str]

PATCH_INFO = {
    'Beauty': f'0.001_Beauty_m1_res.npy',
    'Bosphorus': f'0.001_Bosphorus_m1_res.npy',
    'CityAlley': f'0.001_CityAlley_m1_res.npy',
    'FlowerFocus': f'0.001_FlowerFocus_m1_res.npy',
    'FlowerKids': f'0.001_FlowerKids_m1_res.npy',
    'FlowerPan': f'0.001_FlowerPan_m1_res.npy',
    'HoneyBee': f'0.001_HoneyBee_m1_res.npy',
    'Jockey': f'0.001_Jockey_m1_res.npy',
    'Lips': f'0.001_Lips_m1_res.npy',
    'RaceNight': f'0.001_RaceNight_m1_res.npy',
    'ReadySteadyGo': f'0.001_ReadySteadyGo_m1_res.npy',
    'RiverBank': f'0.001_RiverBank_m1_res.npy',
    'ShakeNDry': f'0.001_ShakeNDry_m1_res.npy',
    'SunBath': f'0.001_SunBath_m1_res.npy',
    'Twilight': f'0.001_Twilight_m1_res.npy',
    'YachtRide': f'0.001_YachtRide_m1_res.npy',
    'Wood': f'0.001_Wood_m1_res.npy',
}


class Strategy(object):
    def __init__(
            self,
            target: int | float = 33,
            tolerance_low: int | float = -5,
            tolerance_high: int | float = 5,
            default_nb_patch: int = 5,
            Kp: float = 1.0,
            Ki: float = 0.0,
            Kd: float = 0.1,
    ):
        self.diff = None
        self.last_ts = None

        self.current_nb_patch = default_nb_patch
        self.target = target
        self.tolerance_low = tolerance_low
        self.tolerance_high = tolerance_high

        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd

        self.integral = 0.0
        self.last_e = 0.0

        self.min_patch = 1
        self.max_patch = 36

    def get_current_nb_patch(self):
        return int(self.current_nb_patch)

    def update(self):
        error = self.target - self.diff

        if self.tolerance_low < error < self.tolerance_high:
            error = 0

        self.integral += error

        derivative = error - self.last_e

        control_signal = self.Kp * error + self.Ki * self.integral + self.Kd * derivative

        self.current_nb_patch += control_signal

        self.current_nb_patch = max(self.min_patch, min(self.current_nb_patch, self.max_patch))

        self.last_e = error

    def update_ts(self, timestamp: int | float):
        if self.last_ts is None:
            self.last_ts = timestamp
            return

        self.diff = timestamp - self.last_ts
        self.last_ts = timestamp

        self.update()

    def update_diff(self, diff: int | float):
        self.diff = diff

        self.update()


def _video_read() -> Iterator[numpy.ndarray]:
    video_filename, (video_width, video_height), video_framerate, (video_format, video_pixel_format) = VIDEOS[
        args.video]
    entry_width = args.src_width
    entry_height = args.src_height

    command = ['ffmpeg', '-y', '-hide_banner', '-loglevel', 'panic']
    if args.real_time:
        command.append('-re')

    if video_format == 'rawvideo':
        command.extend([
            '-f', 'rawvideo',
            '-s', f'{video_width}x{video_height}',
            '-r', f'{video_framerate}',
            '-pix_fmt', f'{video_pixel_format}',
        ])
    command.extend(['-i', str(args.video_path.joinpath(video_filename)), ])
    if video_width != entry_width or video_height != entry_height:
        command.extend(['-vf', f'scale={entry_width}:{entry_height}', ])
    command.extend(['-f', 'rawvideo', '-pix_fmt', 'yuv444p', '-'])

    process = subprocess.Popen(
        args=command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )

    try:
        while True:
            raw_frame = process.stdout.read(entry_width * entry_height * 3)
            if not raw_frame:
                break

            # read the raw frame to numpy
            raw_frame = numpy.frombuffer(raw_frame, dtype=numpy.uint8).reshape((3, entry_height, entry_width))

            yield raw_frame
    finally:
        process.stdout.close()
        process.stderr.close()
        process.wait()


def _read_video_codec() -> Iterator[numpy.ndarray]:
    video_filename, (video_width, video_height), video_framerate, (video_format, video_pixel_format) = VIDEOS[
        args.video]
    entry_width = args.src_width
    entry_height = args.src_height

    # subprocess.run(
    #     args=[
    #         'ffmpeg', '-y', #'-hide_banner', '-loglevel', 'panic',
    #         '-f', 'rawvideo',
    #         '-s', f'{video_width}x{video_height}',
    #         '-r', f'{video_framerate}',
    #         '-pix_fmt', f'{video_pixel_format}',
    #         '-i', str(args.video_path.joinpath(video_filename)),
    #         '-vf', f'scale={entry_width}:{entry_height}',
    #         # TODO: if entry resolution is the original resolution, skip it
    #         # '-f', 'hevc',
    #         '-c:v', 'hevc_nvenc',
    #         '-preset', 'medium',
    #         '-rc', 'constqp', '-qp', f'{args.qp}',
    #         '-f', 'hevc',
    #         'WIP/post-codec.hevc',
    #     ]
    # )

    step_2 = [
        'ffmpeg', '-y', '-hide_banner', '-loglevel', 'panic',
        '-c:v', 'hevc_cuvid',
        '-i', 'WIP/post-codec.hevc',
        '-f', 'rawvideo',
        '-pix_fmt', 'yuv444p',
        'pipe:1'
    ]

    step_2_process = subprocess.Popen(
        args=step_2,
        stdout=subprocess.PIPE,
        text=False,
    )

    try:
        while True:
            raw_frame = step_2_process.stdout.read(entry_width * entry_height * 3)
            if not raw_frame:
                break

            # read the raw frame to numpy
            raw_frame = numpy.frombuffer(raw_frame, dtype=numpy.uint8).reshape((3, entry_height, entry_width))

            yield raw_frame
    finally:
        step_2_process.stdout.close()

        step_2_process.wait()


def video_read() -> Iterator[numpy.ndarray]:
    if args.qp:
        yield from _read_video_codec()
    else:
        yield from _video_read()


def video_read_process(frame_queue: Queue):
    video_filename, (video_width, video_height), video_framerate, (video_format, video_pixel_format) = VIDEOS[
        args.video]
    entry_width = args.src_width
    entry_height = args.src_height

    command = ['ffmpeg', '-y', '-hide_banner', '-loglevel', 'panic']
    if args.real_time:
        command.append('-re')

    if video_format == 'rawvideo':
        command.extend([
            '-f', 'rawvideo',
            '-s', f'{video_width}x{video_height}',
            '-r', f'{video_framerate}',
            '-pix_fmt', f'{video_pixel_format}',
        ])
    command.extend(['-i', str(args.video_path.joinpath(video_filename)), ])
    if video_width != entry_width or video_height != entry_height:
        command.extend(['-vf', f'scale={entry_width}:{entry_height}', ])
    command.extend(['-f', 'rawvideo', '-pix_fmt', 'yuv444p', '-'])

    process = subprocess.Popen(
        args=command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        bufsize=10 ** 9
    )

    try:
        while True:
            raw_frame = process.stdout.read(entry_width * entry_height * 3)
            if not raw_frame:
                break

            # read the raw frame to numpy
            raw_frame = numpy.frombuffer(raw_frame, dtype=numpy.uint8).reshape((3, entry_height, entry_width))

            # yield raw_frame
            frame_queue.put(raw_frame)
    finally:
        process.stdout.close()
        process.stderr.close()
        process.wait()
        frame_queue.close()


def get_display_to_none_process():
    expected_width, expected_height = (args.dst_width, args.dst_height)
    framerate = VIDEOS[args.video][2]

    command = [
        'ffmpeg', '-y', '-hide_banner', '-loglevel', 'panic',
        '-f', 'rawvideo',
        '-s', f'{expected_width}x{expected_height}',
        '-r', f'{framerate}',
        '-pix_fmt', 'yuv444p',
        '-i', 'pipe:0',
        '-f', 'null', '-'
    ]

    process = subprocess.Popen(
        args=command,
        stdin=subprocess.PIPE,
        text=False
    )

    return process


def get_display_process():
    expected_width, expected_height = (args.dst_width, args.dst_height)
    framerate = VIDEOS[args.video][2]

    command = [
        'ffplay', '-hide_banner', '-loglevel', 'panic',
        '-f', 'rawvideo',
        '-video_size', f'{expected_width}x{expected_height}',
        '-framerate', f'{framerate}',
        '-pixel_format', 'yuv444p',
        # '-window_size', '960x540',
        '-vf', 'scale=960:540',
        '-i', '-',
    ]

    process = subprocess.Popen(
        args=command,
        stdin=subprocess.PIPE,
        text=False
    )

    return process


def get_save_for_vmaf_process():
    video_filename, (video_width, video_height), video_framerate, (video_format, video_pixel_format) = VIDEOS[
        args.video]

    expected_width, expected_height = (args.dst_width, args.dst_height)

    command = ['ffmpeg', '-y', '-hide_banner', '-loglevel', 'panic']
    # distorted video
    command.extend([
        '-f', 'rawvideo',
        '-s', f'{expected_width}x{expected_height}',
        '-r', f'{video_framerate}',
        '-pix_fmt', 'yuv444p',
        '-i', 'pipe:0',
    ])

    command.extend([
        '-f', 'rawvideo',
        '-s', f'{video_width}x{video_height}',
        '-r', f'{video_framerate}',
        '-pix_fmt', f'{video_pixel_format}',
        'WIP/WIP.yuv'
    ])

    process = subprocess.Popen(
        args=command,
        stdin=subprocess.PIPE,
        text=False
    )

    return process


def sr_partial_espcn_evasr(
        model,
        frame: numpy.ndarray,
        patch_info: numpy.ndarray,
):
    patch_layout = (args.patch_layout_width, args.patch_layout_height)

    if not (frame.shape[1] % patch_layout[1] == 0 and frame.shape[2] % patch_layout[0] == 0):
        raise ValueError('Wrong frame shape')
    patch_width = frame.shape[2] // patch_layout[0]
    patch_height = frame.shape[1] // patch_layout[1]

    expected_width, expected_height = (args.dst_width, args.dst_height)

    patch_info = patch_info.tolist()

    if args.profiling:
        # read the raw frame to tensor
        with record_function("CPU_TO_GPU"):
            # input_tensor = torch.from_numpy(frame).to(device)
            input_tensor = torch.from_numpy(frame).pin_memory().to(device, non_blocking=True)
        # frame to patches
        with record_function("FRAME_TO_PATCHES"):
            input_tensor = (
                input_tensor
                .unfold(1, patch_height, patch_height)
                .unfold(2, patch_width, patch_width)
                .permute(1, 2, 0, 3, 4)
                .reshape(-1, 3, patch_height, patch_width)
            )

        # ESPCN's NN part
        with record_function("CLONE_NEEDED_PATCHES"):
            input_tensor_y = input_tensor[patch_info, 0:1, :, :].clone()
        with record_function("ESPCN_PRE_PROCESSING"):
            input_tensor_y = input_tensor_y.to(torch.float)
            input_tensor_y = input_tensor_y
            input_tensor_y = input_tensor_y / 255.0
        with record_function("ESPCN"):
            output_tensor_y = model(input_tensor_y)
        with record_function("ESPCN_POST_PROCESSING"):
            output_tensor_y = output_tensor_y
            output_tensor_y = output_tensor_y * 255
            output_tensor_y = output_tensor_y.clamp(0, 255)
            output_tensor_y = output_tensor_y.to(torch.uint8)

        _, _, patch_height_sr, patch_width_sr = output_tensor_y.shape

        # ESPCN's trivial upscale
        with record_function("ESPCN_TRIVIAL_UPSCALE"):
            output_tensor = torchvision.transforms.Resize(
                output_tensor_y.shape[-2:],
                interpolation=torchvision.transforms.InterpolationMode.BICUBIC
            )(input_tensor)
        with record_function("ESPCN_RESULTS_WRITE_BACK"):
            output_tensor[patch_info, 0:1, :, :] = output_tensor_y  # type: torch.Tensor
        with record_function("ESPCN_RESULTS_WRITE_BACK_RESHAPE"):
            output_tensor = (
                output_tensor
                .view(patch_layout[1], patch_layout[0], 3, patch_height_sr, patch_width_sr)
                .permute(2, 0, 3, 1, 4)
                .contiguous()
                .view(3, patch_height_sr * patch_layout[1], patch_width_sr * patch_layout[0])
            )

        # display trivial upscale
        with record_function("DISPLAY_TRIVIAL_UPSCALE"):
            display_tensor = torchvision.transforms.Resize(
                (expected_height, expected_width),
                interpolation=torchvision.transforms.InterpolationMode.NEAREST
            )(output_tensor)

        # read the frame back to CPU and numpy
        with record_function("GPU_TO_CPU"):
            # output_frame = display_tensor.cpu().detach().numpy()
            output_frame = display_tensor.to('cpu', non_blocking=True).detach().numpy()
    else:
        # read the raw frame to tensor
        # input_tensor = torch.from_numpy(frame).to(device)
        input_tensor = torch.from_numpy(frame).pin_memory().to(device, non_blocking=True)

        # frame to patches
        input_tensor = (
            input_tensor
            .unfold(1, patch_height, patch_height)
            .unfold(2, patch_width, patch_width)
            .permute(1, 2, 0, 3, 4)
            .reshape(-1, 3, patch_height, patch_width)
        )

        # ESPCN's NN part
        input_tensor_y = input_tensor[patch_info, 0:1, :, :].clone()
        input_tensor_y = input_tensor_y.to(torch.float)
        input_tensor_y = input_tensor_y
        input_tensor_y = input_tensor_y / 255.0
        output_tensor_y = model(input_tensor_y)
        output_tensor_y = output_tensor_y
        output_tensor_y = output_tensor_y * 255
        output_tensor_y = output_tensor_y.clamp(0, 255)
        output_tensor_y = output_tensor_y.to(torch.uint8)

        _, _, patch_height_sr, patch_width_sr = output_tensor_y.shape

        # ESPCN's trivial upscale
        output_tensor = torchvision.transforms.Resize(
            output_tensor_y.shape[-2:],
            interpolation=torchvision.transforms.InterpolationMode.BICUBIC
        )(input_tensor)
        output_tensor[patch_info, 0:1, :, :] = output_tensor_y  # type: torch.Tensor
        output_tensor = (
            output_tensor
            .view(patch_layout[1], patch_layout[0], 3, patch_height_sr, patch_width_sr)
            .permute(2, 0, 3, 1, 4)
            .contiguous()
            .view(3, patch_height_sr * patch_layout[1], patch_width_sr * patch_layout[0])
        )

        # display trivial upscale
        display_tensor = torchvision.transforms.Resize(
            (expected_height, expected_width),
            interpolation=torchvision.transforms.InterpolationMode.NEAREST
        )(output_tensor)

        # read the frame back to CPU and numpy
        # output_frame = display_tensor.cpu().detach().numpy()
        output_frame = display_tensor.to('cpu', non_blocking=True).detach().numpy()

    return output_frame


def main():
    # load all NN models
    espcn = {}
    for factor in [2, 3, 4, 8]:
        espcn[factor] = torch.jit.load(args.espcn_path.joinpath(ESPCN[factor])).to(device).eval()

    # load patches information
    patch_info = numpy.load(args.patch_path.joinpath(PATCH_INFO[args.video]))
    # sort patches ranking in DEC
    patch_info = numpy.argsort(patch_info, axis=1)[:, ::-1]


    if args.backend == 'none':
        player_process = get_display_to_none_process()
    elif args.backend == 'eval':
        player_process = get_save_for_vmaf_process()
    elif args.backend == 'display':
        player_process = get_display_process()
    else:
        player_process = get_display_to_none_process()

    # start decision
    strategy = Strategy(
        target=args.strategy_target,
        tolerance_low=args.strategy_tolerance_low,
        tolerance_high=args.strategy_tolerance_high,
        default_nb_patch=args.strategy_init_patches,
        Kp=args.strategy_kp,
        Ki=args.strategy_ki,
        Kd=args.strategy_kd,
    )

    logs = []

    if args.profiling:
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], profile_memory=True,
                     record_shapes=True, ) as prof:
            for frame_number, frame in enumerate(video_read()):
                t0 = time.time()
                # send to espcn
                with record_function("OVERALL"):
                    frame_sr = sr_partial_espcn_evasr(
                        espcn[args.scale],
                        frame,
                        patch_info[frame_number, :strategy.get_current_nb_patch()],
                    )
                # strategy.update_diff((time.time() - t0) * 1000.0)

                strategy.update_ts(time.time() * 1000.0)

                player_process.stdin.write(frame_sr.tobytes())

                if args.verbose:
                    print(frame_number, strategy.get_current_nb_patch(), strategy.diff)
                if args.log:
                    logs.append({
                        'frame_number': frame_number,
                        'sred_patches': strategy.get_current_nb_patch(),
                        'execution_time': strategy.diff
                    })

            player_process.stdin.close()
            player_process.communicate()
    else:
        for frame_number, frame in enumerate(video_read()):
            t0 = time.time()
            # send to espcn
            frame_sr = sr_partial_espcn_evasr(
                espcn[args.scale],
                frame,
                patch_info[frame_number, :strategy.get_current_nb_patch()],
            )
            # strategy.update_diff((time.time() - t0) * 1000.0)

            strategy.update_ts(time.time() * 1000.0)

            player_process.stdin.write(frame_sr.tobytes())

            if args.verbose:
                print(frame_number, strategy.get_current_nb_patch(), strategy.diff)
            if args.log:
                logs.append({
                    'frame_number': frame_number,
                    'sred_patches': strategy.get_current_nb_patch(),
                    'execution_time': strategy.diff
                })

        player_process.stdin.close()
        player_process.communicate()

    # frame_queue = Queue(maxsize=120)
    # video_process = Process(target=video_read_process, args=(frame_queue,))
    # video_process.start()
    #
    # if args.profiling:
    #     with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], profile_memory=True,
    #                  record_shapes=True, ) as prof:
    #         frame_number = 0
    #         while True:
    #             try:
    #                 frame = frame_queue.get(timeout=1.0)
    #
    #                 t0 = time.time()
    #                 # send to espcn
    #                 with record_function("OVERALL"):
    #                     frame_sr = sr_partial_espcn_evasr(
    #                         espcn[args.scale],
    #                         frame,
    #                         patch_info[frame_number, :strategy.get_current_nb_patch()],
    #                     )
    #                 # strategy.update_diff((time.time() - t0) * 1000.0)
    #
    #                 strategy.update_ts(time.time() * 1000.0)
    #
    #                 player_process.stdin.write(frame_sr.tobytes())
    #
    #                 if args.verbose:
    #                     print(frame_number, strategy.get_current_nb_patch(), strategy.diff)
    #                 if args.log:
    #                     logs.append({
    #                         'frame_number': frame_number,
    #                         'sred_patches': strategy.get_current_nb_patch(),
    #                         'execution_time': strategy.diff
    #                     })
    #
    #                 frame_number += 1
    #             except ValueError:
    #                 break
    #             except queue.Empty:
    #                 break
    #
    #         player_process.stdin.close()
    #         player_process.communicate()
    # else:
    #     frame_number = 0
    #     while True:
    #         try:
    #             frame = frame_queue.get(timeout=1.0)
    #
    #             t0 = time.time()
    #             # send to espcn
    #             frame_sr = sr_partial_espcn_evasr(
    #                 espcn[args.scale],
    #                 frame,
    #                 patch_info[frame_number, :strategy.get_current_nb_patch()],
    #             )
    #             # strategy.update_diff((time.time() - t0) * 1000.0)
    #
    #             strategy.update_ts(time.time() * 1000.0)
    #
    #             player_process.stdin.write(frame_sr.tobytes())
    #
    #             if args.verbose:
    #                 print(frame_number, strategy.get_current_nb_patch(), strategy.diff)
    #             if args.log:
    #                 logs.append({
    #                     'frame_number': frame_number,
    #                     'sred_patches': strategy.get_current_nb_patch(),
    #                     'execution_time': strategy.diff
    #                 })
    #
    #             frame_number += 1
    #         except ValueError:
    #             break
    #         except queue.Empty:
    #             break
    #
    #     player_process.stdin.close()
    #     player_process.communicate()
    #
    # video_process.join()

    ts = int(time.time() * 1000.0)

    if args.profiling:
        prof.export_chrome_trace(f'profile_{args.video}_{ts}.json')
        with open(f'profile_{args.video}_{ts}.log', 'w') as f:
            f.write(prof.key_averages().table())

    if args.log:
        with open(f'results/set1/log_{args.video}_{ts}.json', 'w') as f:
            json.dump({
                'config': {
                    'video': args.video,
                    'device': args.device,
                    'scale': args.scale,
                    'src_width': args.src_width,
                    'src_height': args.src_height,
                    'dst_width': args.dst_width,
                    'dst_height': args.dst_height,
                    'patch_layout_width': args.patch_layout_width,
                    'patch_layout_height': args.patch_layout_height,
                    'strategy_init_patches': args.strategy_init_patches,
                    'strategy_target': args.strategy_target,
                    'strategy_tolerance_high': args.strategy_tolerance_high,
                    'strategy_tolerance_low': args.strategy_tolerance_low,
                    'strategy_kp': args.strategy_kp,
                    'strategy_ki': args.strategy_ki,
                    'strategy_kd': args.strategy_kd,
                    'backend': args.backend,
                    'real_time': args.real_time,
                    'profiling': args.profiling,
                    'log': args.log,
                    'verbose': args.verbose,
                    'qp': args.qp,
                    'video_path': str(args.video_path),
                    'espcn_path': str(args.espcn_path),
                    'patch_path': str(args.patch_path),
                    'comment': args.comment,
                },
                'logs': logs,
            }, f, indent=4)


if __name__ == '__main__':
    parser = ArgumentParser()

    parser.add_argument('video', choices=VIDEOS_NAME, type=str, help='Name of the testing video.')

    parser.add_argument('--device', type=str, default='cuda', help='Device to use.', choices=['cpu', 'cuda'])
    parser.add_argument('--scale', type=int, default=4, help='Scale factor.', choices=[2, 3, 4, 8])

    parser.add_argument('-sw', '--src_width', type=int, default=960, help='Width of the source video.')
    parser.add_argument('-sh', '--src_height', type=int, default=540, help='Height of the source video.')
    parser.add_argument('-dw', '--dst_width', type=int, default=3840, help='Width of the destination video.')
    parser.add_argument('-dh', '--dst_height', type=int, default=2160, help='Height of the destination video.')
    parser.add_argument('-pyw', '--patch_layout_width', type=int, default=6, help='Number of patches across width')
    parser.add_argument('-pyh', '--patch_layout_height', type=int, default=6, help='Number of patches across height')

    parser.add_argument('-sip', '--strategy_init_patches', type=int, default=5, help='Initial number of patches')
    parser.add_argument('-sta', '--strategy_target', type=int, default=33, help='Target execution time')
    parser.add_argument('-sth', '--strategy_tolerance_high', type=int, default=5, help='Tolerance of the target')
    parser.add_argument('-stl', '--strategy_tolerance_low', type=int, default=-5, help='Tolerance of the target')
    parser.add_argument('-skp', '--strategy_kp', type=float, default=1.0)
    parser.add_argument('-ski', '--strategy_ki', type=float, default=0.0)
    parser.add_argument('-skd', '--strategy_kd', type=float, default=0.1)

    parser.add_argument('--backend', type=str, choices=['none', 'eval', 'display'], default='none',
                        help='Backend to use.')
    parser.add_argument('--real_time', action='store_true', )
    parser.add_argument('--profiling', action='store_true', )
    parser.add_argument('--log', action='store_true', )
    parser.add_argument('-v', '--verbose', action='store_true', )
    parser.add_argument('--qp', type=int, default=None, help='Number of QP.')

    parser.add_argument('--video_path', type=Path, default=Path('/home/Projects/UVG'))
    parser.add_argument('--espcn_path', type=Path,
                        default=Path('/home/Projects/cavs2/super_resolution_build/ESPCN'))
    parser.add_argument('--patch_path', type=Path,
                        default=Path('/home/Projects/cavs2/super_resolution_build/EVASR/saliency_weight'))

    parser.add_argument('--comment', type=str)

    args = parser.parse_args()

    # if args.backend == 'eval':
    #     temp = tempfile.NamedTemporaryFile()
    # else:
    #     temp = None

    if args.device == 'cuda' and torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    main()

    # if args.backend == 'eval':
    #     temp.close()
