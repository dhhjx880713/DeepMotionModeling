import owncloud 
import argparse
import zipfile
import os


CLOUD_URL = "https://cloud.dfki.de/owncloud"


def download_from_oc(user, pw, src_path, dst_path):
    oc = owncloud.Client(CLOUD_URL)
    oc.login(user, pw)
    # oc.get_file(path)
    oc.get_directory_as_zip(src_path, dst_path)
    with zipfile.ZipFile(dst_path, 'r') as zip_file:
        zip_file.extractall(r'./')
    os.remove(dst_path)


def main():
    parser = argparse.ArgumentParser(description='Upload data to owncloud.')
    parser.add_argument('user', nargs='?', help='user')
    parser.add_argument('pw', nargs='?', help='pw')
    parser.add_argument('src_path', nargs='?', default='./workspace/repos/deepmotionmodeling/data', help='path')
    parser.add_argument('dst_path', nargs='?', default=r'./data', help='path')
    args = parser.parse_args()
    if not args.dst_path.endswith(".zip"):
        args.dst_path += ".zip"
    if args.user is not None and args.pw is not None and args.src_path is not None and args.dst_path is not None:
        download_from_oc(args.user, args.pw, args.src_path, args.dst_path)

if __name__ == "__main__":
    main()