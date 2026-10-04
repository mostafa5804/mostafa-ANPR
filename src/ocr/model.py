"""
CRNN Model for Persian License Plate Recognition
معماری: CNN + BiLSTM + CTC
"""

import torch
import torch.nn as nn


class CRNN(nn.Module):
    """
    CRNN برای تشخیص متن پلاک
    ورودی: تصویر خاکستری (1, 32, 160)
    خروجی: توالی کاراکترها (T, batch, num_classes)
    """

    def __init__(self, input_channel=1, num_classes=35, hidden_size=256):
        super(CRNN, self).__init__()

        self.num_classes = num_classes

        # ==================== CNN ====================
        # ورودی: (batch, 1, 32, 160)
        self.cnn = nn.Sequential(
            # Block 1
            nn.Conv2d(input_channel, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # → (64, 16, 80)

            # Block 2
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # → (128, 8, 40)

            # Block 3
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1)),  # → (256, 4, 40)

            # Block 4
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1)),  # → (512, 2, 40)

            # Block 5
            nn.Conv2d(512, 512, kernel_size=2, padding=0),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),  # → (512, 1, 39)
        )

        # ==================== RNN ====================
        self.rnn = nn.Sequential(
            nn.LSTM(
                input_size=512,
                hidden_size=hidden_size,
                num_layers=2,
                bidirectional=True,
                batch_first=False,
                dropout=0.2,
            ),
        )

        # ==================== FC ====================
        self.fc = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, x):
        """
        Args:
            x: (batch, 1, 32, 160)
        Returns:
            output: (T, batch, num_classes)
        """
        # CNN
        conv = self.cnn(x)  # (batch, 512, 1, 39)

        # Reshape برای RNN
        batch, channels, h, w = conv.size()
        conv = conv.squeeze(2)  # (batch, 512, 39)
        conv = conv.permute(2, 0, 1)  # (39, batch, 512)

        # RNN
        rnn_out, _ = self.rnn(conv)  # (39, batch, hidden_size*2)

        # FC
        output = self.fc(rnn_out)  # (39, batch, num_classes)

        return output


def test_model():
    """تست سریع معماری"""
    model = CRNN(input_channel=1, num_classes=35, hidden_size=256)

    # ورودی نمونه
    x = torch.randn(2, 1, 32, 160)
    output = model(x)

    print(f"ورودی: {x.shape}")
    print(f"خروجی: {output.shape}")
    print(f"تعداد پارامترها: {sum(p.numel() for p in model.parameters()):,}")

    # بررسی
    assert output.shape[1] == 2, "batch size اشتباه"
    assert output.shape[2] == 35, "num_classes اشتباه"
    print("✅ تست موفق!")


if __name__ == "__main__":
    test_model()