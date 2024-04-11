import torch
import torch.nn as nn

if __name__ == "__main__":
    """Receiving input and patching"""
    scale = 2
    height = 256
    width = 256
    inp = torch.rand(4, 1, height, width)
    print(f"Original input shape: {inp.shape}")

    inp = torch.reshape(
        inp, (4, 1, scale * scale, height // scale, width // scale))
    print(f"Input shape after applying scale 's': {inp.shape}")

    encoder_input = torch.unbind(inp, dim=2)
    for i in encoder_input:
        print(f"Each input for {scale * scale} encoders has shape: {i.shape}")

    print("-" * 50)

    """Going through encoder"""
    print(
        f"When going through encoder, input has shape: {(4, 1, 128, 128)} turns to: {(4, 1024, 16, 16)}")
    print(f"Then, output is flattened into shape of {4, 1024, 256}")
    print("-" * 50)

    """After having output from CNN encoder"""
    output_list = [torch.rand(4, 1024, 256), torch.rand(
        4, 1024, 256), torch.rand(4, 1024, 256), torch.rand(4, 1024, 256)]

    transformer_input = torch.concat(output_list, dim=1)  # (4, 4096, 256)

    transformer_layer = nn.TransformerEncoderLayer(
        d_model=256, nhead=8, batch_first=True)

    transformer_output = transformer_layer(transformer_input)
    print(f"Output of transformer: {transformer_output.shape}")
    print("-" * 50)

    decoder_input = torch.split(
        transformer_output, split_size_or_sections=1024, dim=1)

    for i in decoder_input:
        print(
            f"After splitting from transformer output, each tensor has shape: {i.shape}")
    print("-" * 50)

    decoder_input_reshape = []

    for i in range(len(decoder_input)):
        decoder_input_reshape.append(torch.reshape(
            decoder_input[i], (4, 1024, 16, 16)))

    for i in decoder_input_reshape:
        print(
            f"After splitting from transformer output and reshaping, each tensor has shape: {i.shape}")
