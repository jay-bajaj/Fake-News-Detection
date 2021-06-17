# 0.75 Marks. 
# To test your trainer and  arePantsonFire class, Just create random tensor and see if everything is working or not.  
from torch.utils.data import DataLoader
from Encoder import Encoder
from Attention import MultiHeadAttention,PositionFeedforward
from trainer import trainer
from LiarLiar import arePantsonFire
from datasets import dataset
from utils import *
from ConvS2S import ConvEncoder



liar_dataset_train=dataset(prep_Data_from='train')
liar_dataset_val=dataset(prep_Data_from='val')
sentence_length,justification_length= liar_dataset_train.get_max_lenghts()
dataloader_train= DataLoader(dataset=liar_dataset_train, batch_size=50)
dataloader_val= DataLoader(dataset=liar_dataset_val, batch_size=25)
statement_encoder=Encoder(conv_layers=5,hidden_dim=512)
justification_encoder=Encoder(conv_layers=5,hidden_dim=512)
multiheadAttention=MultiHeadAttention(hid_dim=512,n_heads=32)
positionFeedForward=PositionFeedforward(hid_dim=512,feedForward_dim=2048)
model=arePantsonFire(statement_encoder,explanation_encoder=justification_encoder,multihead_Attention=multiheadAttention,position_Feedforward=positionFeedForward,hidden_dim=512,max_length_sentence=sentence_length,max_length_justification=justification_length,input_dim=liar_dataset_train.embedding_dim,device='cpu')
# Do not change module_list , otherwise no marks will be awarded
module_list = [liar_dataset_train, liar_dataset_val, dataloader_train, dataloader_val, statement_encoder, justification_encoder, multiheadAttention, positionFeedForward, model]
del  liar_dataset_val, liar_dataset_train, dataloader_train, dataloader_val

trainer(model, module_list[2], module_list[3], num_epochs=1, path_to_save='/home/atharva',checkpoint_path='/home/atharva',checkpoint=100, train_batch=1, test_batch=1, device='cpu')

liar_dataset_test = dataset(prep_Data_from='test')
test_dataloader = DataLoader(dataset=liar_dataset_test, batch_size=1)
infer(model=model, dataloader=test_dataloader)

