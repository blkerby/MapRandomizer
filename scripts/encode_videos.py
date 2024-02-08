from google.cloud import storage
import json
import os

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
    if blob.name not in old_checksums:
        print("New file", blob.name)
    elif old_checksums[blob.name] != blob.md5_hash:
        print("Updated file", blob.name)
    else:
        continue

    base_name = '.'.join(blob.name.split('.')[:-1])
    output_name = base_name + '.webm'
    blob.download_to_filename(tmp_input_file)
    os.system(f"ffmpeg -i {tmp_input_file} -y -vf scale=512:-1:flags=neighbor -pix_fmt yuv420p {tmp_output_file}")
    dst_bucket.blob(output_name).upload_from_filename(tmp_output_file)
    new_checksums[blob.name] = blob.md5_hash

dst_bucket.blob(checksums_blob_name).upload_from_string(json.dumps(new_checksums))
