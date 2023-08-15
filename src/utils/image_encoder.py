import autorootcwd  # noqa: F401
import torch


class ImageEncoder:
    def __init__(self):
        pass

    def encode_labels(self, labels):
        encoded_labels = []
        for label in labels:
            encoded_label = []
            for char in label:
                if char.isdigit():
                    encoded_label.append(int(char))
                else:
                    encoded_label.append(ord(char) - ord('A') + 10)
            encoded_labels.append(torch.tensor(encoded_label))
        return torch.stack(encoded_labels)

    def decode_output(self, output):
        decoded_label = []
        for char_prob in output:
            char_idx = char_prob.argmax().item()
            if char_idx < 10:
                decoded_label.append(str(char_idx))
            else:
                decoded_label.append(chr(char_idx - 10 + ord('A')))
        return ''.join(decoded_label)
