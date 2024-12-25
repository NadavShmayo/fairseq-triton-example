from typing import Dict, List, Optional, Tuple

import torch
from fairseq.models.transformer import TransformerModel
from torch import LongTensor, Tensor

model = TransformerModel.from_pretrained('./wmt14.en-fr.joined-dict.transformer',
                                         'model.pt')


device = 'cuda'


decoder = model.models[0].decoder.eval().to(device)


model.models[0].__class__ = TransformerModel
for decoder_layer in model.models[0].decoder.layers:
    if decoder_layer.need_attn is None:
        decoder_layer.need_attn = False



class WrappedDecoder(torch.nn.Module):
    def __init__(self, decoder):
        super().__init__()
        self.decoder = decoder
        self.layer_incremental_keys: List[Tuple[str, str]] = [(layer.self_attn._incremental_state_id, layer.encoder_attn._incremental_state_id)
                                                              for layer in decoder.layers]
        self.num_self_attn_heads = decoder.layers[0].self_attn.num_heads
        self.num_encoder_attn_heads = decoder.layers[0].encoder_attn.num_heads
        self.self_attn_emb_dim = decoder.layers[0].self_attn.head_dim
        self.encoder_attn_emb_dim = decoder.layers[0].encoder_attn.head_dim


    def _concat_layer_states(self, empty_state):
        pass

    def forward(self, prev_output_tokens: LongTensor,
                encoder_out: Tensor,
                encoder_padding_mask: Tensor,
                incremental_self_attn_states: Tensor,
                incremental_encoder_attn_states: Tensor,
                incremental_encoder_prev_padding_mask: Tensor):
        # incremental_self_attn_states is shaped [num_layers, 2(prevk,prevv), batch_size, num_heads, dest_seq_len, emb_dim]
        # Incremental_encoder_attn_states is shaped [num_layers, 2(prevk,prevv), batch_size, num_heads, src_seq_len, emb_dim]
        # incremental_encoder_prev_padding_mask is shaped [num_layers, batch_size, src_seq_len]
        # Convert incremental states into a nested dictionary
        encoder_out_list = [encoder_out.swapaxes(0, 1)]
        encoder_padding_mask_list = [encoder_padding_mask.to(torch.bool)]
        encoder_out_dict: Dict[str, List[Tensor]] = {
            'encoder_out': encoder_out_list,
            'encoder_padding_mask': encoder_padding_mask_list
        }

        incremental_states_dict: Dict[str, Dict[str, Optional[Tensor]]] = {}
        if prev_output_tokens.shape[1] > 1:

            # Swap axes for triton batching
            incremental_self_attn_states = incremental_self_attn_states.swapaxes(0, 2).contiguous()
            incremental_encoder_attn_states = incremental_encoder_attn_states.swapaxes(0, 2).contiguous()
            incremental_encoder_prev_padding_mask = incremental_encoder_prev_padding_mask.swapaxes(0, 1).contiguous()

            for i, layer_keys in enumerate(self.layer_incremental_keys):
                self_attn_key, encoder_attn_key = layer_keys
                layer_self_attn_states: Dict[str, Optional[Tensor]] = {
                    'prev_key': incremental_self_attn_states[i][0] if incremental_self_attn_states.shape[4] > 0 else None,
                    'prev_value': incremental_self_attn_states[i][1] if incremental_self_attn_states.shape[4] > 0 else None
                }
                layer_encoder_attn_states: Dict[str, Optional[Tensor]] = {
                    'prev_key': incremental_encoder_attn_states[i][0] if incremental_encoder_attn_states.shape[4] > 0 else None,
                    'prev_value': incremental_encoder_attn_states[i][1] if incremental_encoder_attn_states.shape[4] > 0 else None,
                    'prev_key_padding_mask': incremental_encoder_prev_padding_mask[i] if incremental_encoder_prev_padding_mask.shape[2] > 0 else None
                }
                incremental_states_dict[f"{self_attn_key}.attn_state"] = layer_self_attn_states
                incremental_states_dict[f"{encoder_attn_key}.attn_state"] = layer_encoder_attn_states

        decoder_out = self.decoder(prev_output_tokens, encoder_out_dict, incremental_states_dict)

        # decoder adds key and values to incremental_states_dict, appended to the tensor in each layer, we extract the added key and values

        batch_size = prev_output_tokens.size(0)
        device = prev_output_tokens.device
        # Shape: [num_layers, batch_size, num_heads, 1, emb_dim]
        last_token_prev_self_key: Tensor = torch.empty((0, batch_size, self.num_self_attn_heads, 1, self.self_attn_emb_dim), device=device)
        last_token_prev_self_value: Tensor = torch.empty((0, batch_size, self.num_self_attn_heads, 1, self.self_attn_emb_dim), device=device)

        # Shape: [num_layers, batch_size, num_heads, src_seq_len, emb_dim]
        src_seq_len = encoder_out.size(1)
        prev_encoder_keys: Tensor = torch.empty((0, batch_size, self.num_encoder_attn_heads, src_seq_len, self.encoder_attn_emb_dim), device=device)
        prev_encoder_values: Tensor = torch.empty((0, batch_size, self.num_encoder_attn_heads, src_seq_len, self.encoder_attn_emb_dim), device=device)

        # Shape: [num_layers, batch_size, src_seq_len]
        prev_encoder_padding_masks: Tensor = torch.empty((0, batch_size, src_seq_len), device=device)

        for self_attn_key, encoder_attn_key in self.layer_incremental_keys:
            self_attn_key = f"{self_attn_key}.attn_state"
            encoder_attn_key = f"{encoder_attn_key}.attn_state"
            prev_self_key = incremental_states_dict[self_attn_key]['prev_key']
            prev_self_value = incremental_states_dict[self_attn_key]['prev_value']
            prev_encoder_key = incremental_states_dict[encoder_attn_key]['prev_key']
            prev_encoder_value = incremental_states_dict[encoder_attn_key]['prev_value']
            prev_encoder_padding_mask = incremental_states_dict[encoder_attn_key]['prev_key_padding_mask']
            if prev_self_key is not None:
                last_token_prev_self_key = torch.cat([last_token_prev_self_key, prev_self_key[:,:,-1:,:].unsqueeze(0)], dim=0)

            if prev_self_value is not None:
                last_token_prev_self_value = torch.cat([last_token_prev_self_value, prev_self_value[:,:,-1:,:].unsqueeze(0)], dim=0)

            if prev_encoder_key is not None:
                prev_encoder_keys = torch.cat([prev_encoder_keys, prev_encoder_key.unsqueeze(0)], dim=0)

            if prev_encoder_value is not None:
                prev_encoder_values = torch.cat([prev_encoder_values, prev_encoder_value.unsqueeze(0)], dim=0)

            if prev_encoder_padding_mask is not None:
                prev_encoder_padding_masks = torch.cat([prev_encoder_padding_masks, prev_encoder_padding_mask.unsqueeze(0)], dim=0)

        last_token_prev_self_kv = torch.stack([last_token_prev_self_key, last_token_prev_self_value], dim=1)
        prev_encoder_kv = torch.stack([prev_encoder_keys, prev_encoder_values], dim=1)

        # Move batch axis to be first axis for triton
        last_token_prev_self_kv = last_token_prev_self_kv.swapaxes(0, 2)
        prev_encoder_kv = prev_encoder_kv.swapaxes(0, 2)
        prev_encoder_padding_masks = prev_encoder_padding_masks.swapaxes(0, 1)

        attn = decoder_out[1]['attn'][0]
        if attn is not None:
            attn = attn.contiguous().cuda()

        logits, attn = decoder_out[0].contiguous().cuda(), attn
        return logits, attn, last_token_prev_self_kv.contiguous(), prev_encoder_kv.contiguous(), prev_encoder_padding_masks.contiguous()



tokens = torch.LongTensor([[2, 2, 3, 4, 5, 6, 7, 8, 9, 10], [2, 2, 3, 4, 5, 6, 7, 8, 9, 10], [2, 2, 3, 4, 5, 6, 7, 8, 9, 10]]).to(device)
lengths = torch.LongTensor([[10, 10, 10]]).to(device)
encoder = model.models[0].encoder.eval().to(device)
encoder_outs = encoder(tokens, lengths)


encoder_outs['encoder_out'][0] = encoder_outs['encoder_out'][0].swapaxes(0, 1)


wrapped_decoder = WrappedDecoder(decoder)

incremental_self_attn_states = torch.zeros((3, 2, 6, 16, 2, 64)).to(device)
incremental_enecoder_attn_states = torch.zeros((3, 2, 6, 16, 10, 64)).to(device)
incremental_encoder_prev_padding_mask = torch.zeros((3, 6, 10)).to(device)
incremental_states_dict = {
    'incremental_self_attn_states': incremental_self_attn_states,
    'incremental_encoder_attn_states': incremental_enecoder_attn_states,
    'incremental_encoder_prev_padding_mask': incremental_encoder_prev_padding_mask
}

res = wrapped_decoder(torch.LongTensor([[2, 2], [2, 2], [2, 2]]).to(device), encoder_outs['encoder_out'][0], encoder_outs['encoder_padding_mask'][0], **incremental_states_dict)
scripted = torch.jit.script(wrapped_decoder)
res = scripted(torch.LongTensor([[2, 2], [2, 2], [2, 2]]).to(device), encoder_outs['encoder_out'][0], encoder_outs['encoder_padding_mask'][0], **incremental_states_dict)
print(res[0].shape)
print(res[1].shape)
print(res[2].shape)
print(res[3].shape)
print(res[4].shape)
print(len(res))
scripted.save('model.pt')
