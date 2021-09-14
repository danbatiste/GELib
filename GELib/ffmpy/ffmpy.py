import os


def ffmpy(output_file_name, prefix="frame", path=os.getcwd(), fps=60):
    """Runs ffmpeg to stitch together frames in a given folder"""
    # Travel to working directory
    original_path = os.getcwd()
    os.chdir(path)

    # Get frame info
    frames = [file for file in os.listdir() if file[:len(prefix)] == prefix]
    frame_count = len(frames)
    zeros = len(str(frame_count))
    print(frames)
    input_format = frames[0].split('.')[-1]

    # Run ffmpeg
    command = f"ffmpeg -framerate {fps} -start_number 0000 -i frame%0{zeros}d.{input_format} {output_file_name}"
    os.system(command)

    # Return to original path
    os.chdir(original_path)


if __name__ == "__main__":
    ffmpy("test.mp4", path="ffmpy")