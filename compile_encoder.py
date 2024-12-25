import torch
from fairseq.models.transformer import TransformerModel
model = TransformerModel.from_pretrained('./wmt14.en-fr.joined-dict.transformer',
                                         'model.pt')

device = 'cuda'

encoder = model.models[0].encoder.eval().to(device)


# This is for triton, with swapping the batch axis
class WrappedEncoder(torch.nn.Module):
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder

    def forward(self, tokens, lengths):
        output = self.encoder(tokens, lengths)
        # Convert padding mask to int because dlpack doesn't support bool
        return output['encoder_out'][0].permute(1, 0, 2).contiguous().to(tokens.device), output['encoder_padding_mask'][0].to(torch.int64).contiguous().to(tokens.device)



wrapped_encoder = WrappedEncoder(encoder)
res = encoder(torch.LongTensor([[2, 2]]).to(device), torch.LongTensor([[2]]).to(device))
res = wrapped_encoder(torch.LongTensor([[2, 2, 2], [2, 2, 2]]).to(device), torch.LongTensor([[3], [3]]).to(device))
print(res[0].shape)
scripted = torch.jit.script(wrapped_encoder)
scripted.save('model.pt')
