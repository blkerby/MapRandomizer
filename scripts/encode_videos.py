from google.cloud import storage
import json
import os

tmp_input_prefix = "/tmp/video"
tmp_input_listing = "/tmp/video_listing.txt"
tmp_input_file = "/tmp/video.avi"
tmp_output_file = "/tmp/video.webm"
checksums_blob_name = 'checksums.json'

client = storage.Client(project="super-metroid-map-randomizer")
src_bucket = client.get_bucket('super-metroid-map-rando-videos')
dst_bucket = client.get_bucket('super-metroid-map-rando-videos-webm')

old_checksums = json.loads(dst_bucket.blob(checksums_blob_name).download_as_string())
# old_checksums = {}
new_checksums = {**old_checksums}

for blob in src_bucket.list_blobs():
    base_name = '.'.join(blob.name.split('.')[:-1])
    if base_name[-2] == '_' and '1' <= base_name[-1] <= '9':
        continue
    
    if blob.name not in old_checksums:
        print("New file", blob.name)
    elif old_checksums[blob.name] != blob.md5_hash:
        print("Updated file", blob.name)
    else:
        continue

    # For videos that are broken into multiple files, collect all their parts:
    extension = blob.name.split('.')[-1]
    all_blobs = [blob]
    i = 1
    while True:
        next_name = f"{base_name}_{i}.{extension}"
        result = src_bucket.get_blob(next_name)
        if result is None:
            break;
        all_blobs.append(result)
        i = i + 1

    if len(all_blobs) > 1:
        tmp_filenames = []
        for i, b in enumerate(all_blobs):
            filename = f"{tmp_input_prefix}_{i}.{extension}"
            b.download_to_filename(filename)
            tmp_filenames.append(filename)
        listing_content = "\n".join("file {}".format(f) for f in tmp_filenames)
        listing_file = open(tmp_input_listing, "w")
        listing_file.write(listing_content)
        listing_file.close()

        print("Concatenating files: ", tmp_filenames)
        os.system(f"ffmpeg -y -f concat -safe 0 -i {tmp_input_listing} -c copy {tmp_input_file}")
    else:
        blob.download_to_filename(tmp_input_file)

    output_name = base_name + '.webm'
    os.system(f"ffmpeg -i {tmp_input_file} -y -vf scale=512:-1:flags=neighbor -pix_fmt yuv420p {tmp_output_file}")
    dst_bucket.blob(output_name).upload_from_filename(tmp_output_file)
    new_checksums[blob.name] = blob.md5_hash

dst_bucket.blob(checksums_blob_name).upload_from_string(json.dumps(new_checksums))
