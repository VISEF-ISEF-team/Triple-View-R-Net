from keras.layers import Input, Conv2D, MaxPooling2D, Dropout, Conv2DTranspose, concatenate, Softmax
from keras.models import Model

class UNetModel:
    def __init__(self, cf):
        self.cf = cf
        self.encoder_conv = []
        self.decoder_conv = []
        self.inputs = Input((cf['h'], cf['w'], cf['c']))
        
    def encoder(self):
        x = self.inputs
        # Encoder
        c1 = Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(x)
        c1 = Dropout(0.1)(c1)
        c1 = Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c1)
        p1 = MaxPooling2D((2, 2))(c1)
        
        c2 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p1)
        c2 = Dropout(0.1)(c2)
        c2 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c2)
        p2 = MaxPooling2D((2, 2))(c2)
         
        c3 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p2)
        c3 = Dropout(0.2)(c3)
        c3 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c3)
        p3 = MaxPooling2D((2, 2))(c3)
         
        c4 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p3)
        c4 = Dropout(0.2)(c4)
        c4 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c4)
        p4 = MaxPooling2D(pool_size=(2, 2))(c4)
         
        c5 = Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p4)
        c5 = Dropout(0.3)(c5)
        c5 = Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c5)
        
        self.encoder_conv = [c1, c2, c3, c4, c5]
        return c5
    
    def decoder(self, encoded):
        # Decoder
        c6 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(encoded)
        c6 = concatenate([c6, self.encoder_conv[3]])
        c6 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c6)
        c6 = Dropout(0.2)(c6)
        c6 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c6)
         
        c7 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c6)
        c7 = concatenate([c7, self.encoder_conv[2]])
        c7 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c7)
        c7 = Dropout(0.2)(c7)
        c7 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c7)
         
        c8 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(c7)
        c8 = concatenate([c8, self.encoder_conv[1]])
        c8 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c8)
        c8 = Dropout(0.1)(c8)
        c8 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c8)
         
        c9 = Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same')(c8)
        c9 = concatenate([c9, self.encoder_conv[0]], axis=3)
        c9 = Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c9)
        c9 = Dropout(0.1)(c9)
        c9 = Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c9)
        
        self.decoder_conv = [c6, c7, c8, c9]
        return c9

    def build_model(self):
        encoded = self.encoder()
        decoded = self.decoder(encoded)
        outputs = Conv2D(self.cf['n'], (1, 1), activation='softmax')(decoded)
        model = Model(inputs=self.inputs, outputs=outputs)
        return model
