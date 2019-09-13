
# char_vector = "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ-'.!?,\""
char_vector = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ-'.!?,\""
num_classes = len(char_vector) + 1
# batch_size = 64
batch_size = 64
max_word_len = 16
img_w = 100
img_h = 31



# Traceback (most recent call last):
#   File "C:\Users\Admin\Anaconda3\lib\site-packages\PIL\ImageFile.py", line 103,
# in __init__
#     self._open()
#   File "C:\Users\Admin\Anaconda3\lib\site-packages\PIL\JpegImagePlugin.py", line
#  326, in _open
#     i = i8(s)
#   File "C:\Users\Admin\Anaconda3\lib\site-packages\PIL\_binary.py", line 19, in
# i8
#     return c if c.__class__ is int else c[0]
# IndexError: index out of range

# During handling of the above exception, another exception occurred:

# Traceback (most recent call last):
#   File "crnn.py", line 88, in <module>
#     x_batch, y_batch, dt_batch = gen.__next__()
#   File "C:\Users\Admin\Desktop\Synth 90k\mjsynth\mnt\ramdisk\max\90kDICT32px\dat
# a.py", line 32, in data_generator
#     img = imread(path + img_path)[:, :, 0]
#   File "C:\Users\Admin\Anaconda3\lib\site-packages\imageio\core\functions.py", l
# ine 221, in imread
#     reader = read(uri, format, "i", **kwargs)
#   File "C:\Users\Admin\Anaconda3\lib\site-packages\imageio\core\functions.py", l
# ine 143, in get_reader
#     return format.get_reader(request)
#   File "C:\Users\Admin\Anaconda3\lib\site-packages\imageio\core\format.py", line
#  164, in get_reader
#     return self.Reader(self, request)
#   File "C:\Users\Admin\Anaconda3\lib\site-packages\imageio\core\format.py", line
#  214, in __init__
#     self._open(**self.request.kwargs.copy())
#   File "C:\Users\Admin\Anaconda3\lib\site-packages\imageio\plugins\pillow.py", l
# ine 423, in _open
#     return PillowFormat.Reader._open(self, pilmode=pilmode, as_gray=as_gray)
#   File "C:\Users\Admin\Anaconda3\lib\site-packages\imageio\plugins\pillow.py", l
# ine 127, in _open
#     self._im = factory(self._fp, "")
#   File "C:\Users\Admin\Anaconda3\lib\site-packages\PIL\JpegImagePlugin.py", line
#  779, in jpeg_factory
#     im = JpegImageFile(fp, filename)
#   File "C:\Users\Admin\Anaconda3\lib\site-packages\PIL\ImageFile.py", line 112,
# in __init__
#     raise SyntaxError(v)
# SyntaxError: index out of range

