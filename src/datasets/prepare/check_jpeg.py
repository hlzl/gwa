###############
# NOTE: Sometimes JPEG files do not acutally have JPEG headers.
# As this breaks the efficient tf.image.decode_and_crop_jpeg(...),
# we can easily check a directory (and subdirectories) with this script.
###############
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from PIL import Image


def is_jpeg(file_path):
    try:
        with open(file_path, "rb") as f:
            header = f.read(2)
        return header == b"\xff\xd8"  # JPEG magic number
    except Exception:
        return False


def check_file(file_path):
    if not is_jpeg(file_path):
        return file_path  # Return file name if it's not a JPEG
    return None


def check_directory(directory, max_threads=8):
    all_files = [
        os.path.join(root, file) for root, _, files in os.walk(directory) for file in files
    ]
    non_jpeg_files = []

    with ThreadPoolExecutor(max_threads) as executor:
        futures = {executor.submit(check_file, file): file for file in all_files}

        # Use tqdm to track progress
        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing files"):
            result = future.result()
            if result:
                non_jpeg_files.append(result)

    # Output all non-JPEG files
    print("\nNon-JPEG Files:")
    for file in non_jpeg_files:
        print(file)

    return non_jpeg_files


def convert_to_jpeg(non_jpeg_files):
    for file_path in non_jpeg_files:
        try:
            # Open the file
            img = Image.open(file_path)
            # Convert to RGB (ensures compatibility for JPEG)
            img = img.convert("RGB")
            # Save the image as JPEG with the same name
            new_file_path = os.path.splitext(file_path)[0] + ".JPEG"
            img.save(new_file_path, "JPEG")
            print(f"Converted and saved: {new_file_path}")
        except Exception as e:
            print(f"Failed to convert {file_path}: {e}")


print("Script started.")
directory = "/path/to/ILSVRC2012/train"
non_jpeg_files = check_directory(directory)

# Again for validation - likely does not have any corrupted files
print("Train set done.")
directory = "/path/to/ILSVRC2012/val"
non_jpeg_files = check_directory(directory)
