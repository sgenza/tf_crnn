# Model configuration
char_vector = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ-'.!?,\""
num_classes = len(char_vector) + 1
batch_size = 64
max_word_len = 16
img_w = 100
img_h = 31
lstm_units = 256
synth_len = 7224612
svt_test_len = 646
IIIT5K_test_len = 3000