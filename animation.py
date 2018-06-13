from PIL import Image, ImageFilter
import pathlib
import argparse

# パーサーを作る
parser = argparse.ArgumentParser(
    prog='animation.py',  # プログラム名
    usage='Make animation of generated images by GIF',  # プログラムの利用方法
    description='description',  # 引数のヘルプの前に表示
    epilog='end',  # 引数のヘルプの後で表示
    add_help=True,  # -h/–help オプションの追加
)

# 引数の追加
parser.add_argument('-s', '--seed', help='seed',
                    type=int, required=True)
parser.add_argument('-n', '--number', help='the number of experiments',
                    type=int, required=True)
parser.add_argument('-e', '--epoch', help='the number of epoch, defalut value is 100',
                    type=int, default=100)
parser.add_argument('-V', '--version', version='%(prog)s 1.0.0',
                    action='version',
                    default=False)

# 引数を解析する
args = parser.parse_args()

number = args.number  # nmber of experiments
seed = args.seed  # seed
strings = "{0}_{1}".format(number, seed)
# Pillow のGIF生成，画像読み込みは以下のサイトを参照
# https://note.nkmk.me/python-pillow-gif/
# https://note.nkmk.me/python-pillow-basic/
#path = pathlib.Path("result_{}/preview".format(strings))
path = pathlib.Path("./result_{0}/result_{1}/preview".format(number, strings))

# store image to use as frame to array "imgs"
imgs = []
for epoch in range(1, args.epoch + 1):
    img = Image.open(path / "{}epoch.jpg".format(epoch))
    imgs.append(img)

# make gif
imgs[0].save('result_{0}/result_{1}/anim_{1}.gif'.format(number, strings),
             save_all=True, append_images=imgs[1:],
             optimize=False, duration=200, loop=0)
for img in imgs:
    img.close()
