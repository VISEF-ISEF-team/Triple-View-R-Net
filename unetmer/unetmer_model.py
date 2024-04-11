import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from einops import repeat, rearrange
from torchinfo import summary
import gc


class SingleConvBlock(nn.Module):
    def __init__(self, inc, outc):
        super().__init__()
        self.single_conv_block = nn.Sequential(
            nn.Conv2d(
                in_channels=inc, out_channels=outc, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x = self.single_conv_block(x)
        x = self.pool(x)
        return x


class SingleDeconvBlock(nn.Module):
    def __init__(self, inc, outc) -> None:
        super().__init__()
        self.single_conv_block = nn.Sequential(
            nn.Conv2d(
                in_channels=inc, out_channels=outc, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.single_conv_block(x)
        return x


class EncoderBlock(nn.Module):
    def __init__(self, in_channel=1, features=[64, 128, 256, 512]):
        super().__init__()
        self.features = features
        self.input_conv = nn.Conv2d(
            in_channels=in_channel, out_channels=features[0], kernel_size=3, stride=1, padding=1)

        self.encoder_list = nn.ModuleList()
        self.encoder_list.append(self.input_conv)

        """Initialize all necessary encoders"""
        for i in range(0, len(self.features) - 1):
            single_block = SingleConvBlock(
                inc=self.features[i], outc=self.features[i + 1])
            self.encoder_list.append(single_block)

        self.skip_connections = []

    def forward(self, x):
        """Deprecated"""
        # x = self.input_conv(x)
        # self.skip_connections.append(x)
        # for i in range(0, len(self.features) - 1):
        #     single_block = SingleConvBlock(
        #         inc=self.features[i], outc=self.features[i + 1])
        #     x = single_block(x)

        """New"""
        self.skip_connections.clear()
        for i in range(len(self.encoder_list)):
            encoder = self.encoder_list[i]
            x = encoder(x)

            self.skip_connections.append(x)

        return x, self.skip_connections


class DecoderBlock(nn.Module):
    def __init__(self, features=[512, 256, 128, 64]):
        super().__init__()
        self.features = features
        self.skip_connection_counter = 0
        self.decoder_list = nn.ModuleList()

        for i in range(len(self.features)):
            transpose_layer = nn.ConvTranspose2d(
                in_channels=self.features[i] * 2, out_channels=self.features[i], kernel_size=2, stride=2)
            decoder_layer = SingleDeconvBlock(
                inc=self.features[i] * 2, outc=self.features[i])

            self.decoder_list.append(transpose_layer)
            self.decoder_list.append(decoder_layer)

    def forward(self, x, skip_connections):
        """Deprecated"""
        # for i in range(0, len(self.features)):
        #     # pass through deconv layer
        #     x = nn.ConvTranspose2d(
        #         in_channels=self.features[i] * 2, out_channels=self.features[i], kernel_size=2, stride=2)(x)

        #     # get skip connection
        #     skip_connection_input = skip_connections[self.skip_connection_counter]

        #     # concat with skip connection
        #     x = torch.cat((x, skip_connection_input), dim=1)

        #     # get single deconv block
        #     single_block = SingleDeconvBlock(
        #         inc=self.features[i] * 2, outc=self.features[i])

        #     # increase skip connection counter
        #     self.skip_connection_counter += 1

        #     # do forwad pass
        #     x = single_block(x)

        """New"""
        self.skip_connection_counter = 0
        for i in range(0, len(self.decoder_list), 2):
            # get transpose and deconv layer
            transpose_layer = self.decoder_list[i]
            single_deconv_block = self.decoder_list[i + 1]

            # pass through transpose
            x = transpose_layer(x)

            # get skip connection
            skip_connection_input = skip_connections[self.skip_connection_counter]

            # concat with skip connection
            x = torch.cat((x, skip_connection_input), dim=1)

            # pass through deconv block
            x = single_deconv_block(x)

            # increase skip connection counter
            self.skip_connection_counter += 1

        return x


class AbsPositionalEncoding1D(nn.Module):
    def __init__(self, tokens, dim):
        super().__init__()
        self.abs_pos_enc = nn.Parameter(torch.randn(1, tokens, dim))

    def forward(self, x):
        batch = x.size()[0]

        return x + repeat(self.abs_pos_enc, 'b ... -> (b tile) ...', tile=batch // self.abs_pos_enc.shape[0])


class Unetmer(nn.Module):

    def __init__(self, inc=1, outc=8, scale=2, original_size=256):
        super().__init__()

        """Define other params"""
        self.original_size = original_size
        self.scale = scale
        self.num_encode_decode_sequence = self.scale * self.scale  # 4
        self.patched_size = int(original_size / self.scale)  # 128
        self.d_model = int((self.patched_size /
                            (2 ** self.num_encode_decode_sequence)) * (self.patched_size / (2 ** self.num_encode_decode_sequence)))  # 64 || dim model = (size / 2^4) * (size / 2^4)\

        """Define layers"""
        self.abs_pos_encoding = AbsPositionalEncoding1D(4096, self.d_model)
        self.transformer = nn.TransformerEncoderLayer(
            d_model=self.d_model, nhead=8, batch_first=True)

        self.encoder_blocks = nn.ModuleList()
        self.decoder_blocks = nn.ModuleList()

        self.skip_connections_list = []

        self.outputs = []
        self.out_conv = nn.Conv2d(
            in_channels=64, out_channels=outc, kernel_size=3, stride=1, padding=1)

        for i in range(self.num_encode_decode_sequence):
            single_encoder_block = EncoderBlock(inc)
            self.encoder_blocks.append(single_encoder_block)
            self.encoder_blocks.append(SingleConvBlock(inc=512, outc=1024))

            single_decoder_block = DecoderBlock()
            self.decoder_blocks.append(single_decoder_block)

    def split_inputs(self, inputs):
        """
        inputs should be of shape (batch, channel, size, size) 
        """
        splitted_image_list = []

        for i in range(self.scale):
            for j in range(self.scale):
                splitted_image = inputs[:, :, i * self.patched_size:(i + 1) *
                                        self.patched_size, j * self.patched_size:(j + 1) * self.patched_size]
                splitted_image_list.append(splitted_image)

        return splitted_image_list

    def join_outputs(self, outputs):
        y = outputs.shape[1]
        new_output = outputs.reshape(
            y, 8, self.original_size, self.original_size)

        return new_output

    def forward(self, original_inputs):
        self.skip_connections_list.clear()

        """Convert inputs to list of inputs"""
        inputs = self.split_inputs(original_inputs)

        """Pass through encoder"""
        encoder_block_counter = 0
        for i in range(len(inputs)):
            x = inputs[i]

            # pass through main encoder block
            x, skip_connections = self.encoder_blocks[encoder_block_counter](x)
            skip_connections = skip_connections[::-1]
            self.skip_connections_list.append(skip_connections)

            # pass through bottle neck layer
            x = self.encoder_blocks[encoder_block_counter + 1](x)

            # print(f"After passing through encoder: {x.shape}")

            encoder_block_counter += 2

            inputs[i] = x

        # print(f"Length skip connection: {len(self.skip_connections_list)}")

        """Pass through Transformers"""
        # flatten to 1D sequence
        for i in range(len(inputs)):
            inputs[i] = inputs[i].flatten(start_dim=2)

        # concat to sequence
        transformer_input = torch.concat(inputs, dim=1)
        del inputs

        # abs positional encoding
        transformer_input = self.abs_pos_encoding(transformer_input)

        # pass through transformer
        transformer_output = self.transformer(transformer_input)
        del transformer_input

        # print(f"Transformer output: {transformer_output.shape}")

        decoder_input = torch.split(
            transformer_output, split_size_or_sections=1024, dim=1)
        decoder_input = list(decoder_input)
        del transformer_output

        gc.collect()

        for i in range(len(decoder_input)):
            # calculate last dimension
            new_shape_last_dim = int(decoder_input[i].shape[-1] ** 0.5)

            # calculate new tensor shape
            new_shape = (
                decoder_input[i].shape[0], decoder_input[i].shape[1], new_shape_last_dim, new_shape_last_dim)

            # reassign decoder input
            decoder_input[i] = torch.reshape(decoder_input[i], new_shape)

        # for i in decoder_input:
        #     print(f"Decoder input: {i.shape}")

        """Pass through decoder"""
        decoder_block_counter = 0
        for i in range(len(decoder_input)):
            x = decoder_input[i]

            # pass through decoder
            x = self.decoder_blocks[decoder_block_counter](
                x, self.skip_connections_list[i])

            # print(f"After passing through decoder: {x.shape}")

            x = self.out_conv(x)
            decoder_block_counter += 1

            self.outputs.append(x)

        self.outputs_torch = torch.stack(self.outputs)
        self.outputs_torch = self.join_outputs(self.outputs_torch)
        # print(f"Final model output shape: {self.outputs.shape}")
        self.outputs.clear()
        return self.outputs_torch


if __name__ == "__main__":
    unetmer = Unetmer(original_size=256)

    # inputs have not been splitted
    normal_input = torch.rand(4, 1, 256, 256)

    outputs = unetmer(normal_input)
    print("-" * 50)

    for i in outputs:
        print(f"Model output: {i.shape}")

    print(f"Output received: {outputs.shape}")

    normal_input_2 = torch.rand(4, 1, 256, 256)
    outputs2 = unetmer(normal_input_2)
    print(f"Second output received: {outputs2.shape} ")

    summary(unetmer, (4, 1, 256, 256))
