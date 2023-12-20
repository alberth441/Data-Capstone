from tensorflow import keras

MODEL_PATH = './model_sunda.h5'
MODEL = keras.models.load_model(MODEL_PATH)

CLASS_NAMES = ['sunda_a', 'sunda_ae', 'sunda_ba', 'sunda_ca', 'sunda_da', 'sunda_e', 'sunda_eu', 'sunda_fa', 'sunda_ga', 'sunda_ha', 'sunda_i','sunda_ja', 'sunda_ka', 'sunda_la', 'sunda_ma', 'sunda_na', 'sunda_nga', 'sunda_nya', 'sunda_o', 
          'sunda_pa', 'sunda_qa', 'sunda_ra', 'sunda_sa', 'sunda_ta', 'sunda_u', 'sunda_va', 'sunda_wa', 'sunda_xa', 'sunda_ya', 'sunda_za']