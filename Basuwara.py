from tensorflow import keras

MODEL_PATH = './api_model/save_model/model.h5'
MODEL = keras.models.load_model(MODEL_PATH)

CLASS_NAMES = ['bali_ba', 'bali_ca', 'bali_da', 'bali_ga', 'bali_ha', 'bali_ja', 'bali_ka', 'bali_la', 'bali_ma', 'bali_na', 'bali_nga', 'bali_nya', 'bali_pa', 'bali_ra', 'bali_sa', 'bali_ta', 'bali_wa', 'bali_ya',
          'jawa_ba', 'jawa_ca', 'jawa_da', 'jawa_dha', 'jawa_ga', 'jawa_ha', 'jawa_ja', 'jawa_ka', 'jawa_la', 'jawa_ma', 'jawa_na', 'jawa_nga', 'jawa_nya', 'jawa_pa', 'jawa_ra', 'jawa_sa', 'jawa_ta', 'jawa_tha', 'jawa_wa', 'jawa_ya',
          'sunda_a', 'sunda_ae', 'sunda_ba', 'sunda_ca', 'sunda_da', 'sunda_e', 'sunda_eu', 'sunda_fa', 'sunda_ga', 'sunda_ha', 'sunda_i','sunda_ja', 'sunda_ka', 'sunda_la', 'sunda_ma', 'sunda_na', 'sunda_nga', 'sunda_nya', 'sunda_o', 
          'sunda_pa', 'sunda_qa', 'sunda_ra', 'sunda_sa', 'sunda_ta', 'sunda_u', 'sunda_va', 'sunda_wa', 'sunda_xa', 'sunda_ya', 'sunda_za']