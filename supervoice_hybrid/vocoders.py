from encodec import EncodecModel

def load_encodec_encoder():
    encodec_model = EncodecModel.encodec_model_24khz()
    encodec_model.set_target_bandwidth(6.0)
    return encodec_model

def load_encodec_decoder_direct():
    encodec_model = EncodecModel.encodec_model_24khz()
    encodec_model.set_target_bandwidth(6.0)
    return encodec_model