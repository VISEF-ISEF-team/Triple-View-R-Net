import ml_collections


def get_config():
    config = ml_collections.ConfigDict()

    # Input
    config.num_classes = 14
    config.in_channels = 1
    config.img_size = 256
    config.p = [16, 8, 4, 2]

    # Rotatory attention
    config.df = [64, 128, 256, 512]
    config.dk = [64, 128, 256, 512]
    config.dv = [64, 128, 256, 512]
    config.dq = [64, 128, 256, 512]

    # Channel-wise Transformer
    config.patches_size = [16, 8, 4, 2]
    config.channel_nums = [64, 128, 256, 512]
    config.mlp_ratio = 4
    config.KV_size = 64 + 128 + 256 + 512

    # Transformer
    config.transformer = ml_collections.ConfigDict()
    config.transformer.num_heads = 12
    config.transformer.num_layers = 6
    config.transformer.att_dropout_rate = 0.0
    config.transformer.dropout_rate = 0.1

    # Resnet hybid model
    config.resnet = ml_collections.ConfigDict()
    config.resnet.num_layers = (3, 4, 9)
    config.resnet.width_factor = 1

    # Decoder
    config.classifier = 'seg'
    config.n_skip = 3
    config.skip_channels = [512, 256, 64, 16]
    config.decoder_channels = (256, 128, 64, 16)

    # Others
    config.activation = 'softmax'
    config.vis = True

    return config
