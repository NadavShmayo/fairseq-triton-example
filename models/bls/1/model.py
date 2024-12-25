import argparse
from typing import Dict, List

import torch
from fairseq.models import FairseqEncoder, FairseqIncrementalDecoder
from fairseq.models.transformer import TransformerModel
from fairseq.sequence_generator import SequenceGenerator, EnsembleModel
from fairseq.tasks import FairseqTask
from torch import Tensor


from fairseq.models.transformer import TransformerModel

import torch
from torch import Tensor
from typing import Dict, Optional


import triton_python_backend_utils as pb_utils
import json
from torch.utils.dlpack import from_dlpack, to_dlpack
import numpy as np


def get_torch_from_request(request, input_name: str) -> Tensor:
    return from_dlpack(pb_utils.get_input_tensor_by_name(request, input_name).to_dlpack())


def get_torch_from_response(response, output_name: str) -> Tensor:
    return from_dlpack(pb_utils.get_output_tensor_by_name(response, output_name).to_dlpack()).cuda()



def reorder_incremental_state(
        incremental_state: Dict[str, Tensor],
        new_order: Tensor
):
    """
    Reorder the incremental state for all attention types.

    Args:
        incremental_state (Dict[str, Tensor]): The current incremental state.
        new_order (Tensor): The new order of the batch dimension.

    Returns:
        Dict[str, Tensor]: The reordered incremental state.
    """
    if 'incremental_self_attn_states' in incremental_state:
        incremental_state['incremental_self_attn_states'] = reorder_self_attn_states(
            incremental_state['incremental_self_attn_states'], new_order
        )

    if 'incremental_encoder_attn_states' in incremental_state:
        incremental_state['incremental_encoder_attn_states'] = reorder_encoder_attn_states(
            incremental_state['incremental_encoder_attn_states'], new_order
        )

    if 'incremental_encoder_prev_padding_mask' in incremental_state:
        incremental_state['incremental_encoder_prev_padding_mask'] = reorder_encoder_padding_mask(
            incremental_state['incremental_encoder_prev_padding_mask'], new_order
        )


def reorder_self_attn_states(states: Tensor, new_order: Tensor) -> Tensor:
    """
    Reorder the self-attention incremental states.

    Shape:
        - states: [batch, 2, num_layers, num_heads, dst_seq_len, head_dim]
        - new_order: [new_batch_size]
        - output: [new_batch_size, 2, num_layers, num_heads, dst_seq_len, head_dim]
    """
    return states.index_select(0, new_order)

def reorder_encoder_attn_states(states: Tensor, new_order: Tensor) -> Tensor:
    """
    Reorder the encoder-decoder attention incremental states.

    Shape:
        - states: [num_layers, 2, batch, num_heads, src_seq_len, head_dim]
        - new_order: [new_batch_size]
        - output: [num_layers, 2, new_batch_size, num_heads, src_seq_len, head_dim]
    """
    return states.index_select(0, new_order)


def reorder_encoder_padding_mask(padding_mask: Tensor, new_order: Tensor) -> Tensor:
    """
    Reorder the encoder padding mask.

    Shape:
        - padding_mask: [batch, num_layers, src_seq_len]
        - new_order: [new_batch_size]
        - output: [new_batch_size, num_layers, src_seq_len]
    """
    return padding_mask.index_select(0, new_order)


class TritonEncoder(FairseqEncoder):
    def __init__(self):
        super().__init__(None)

    def forward(self, src_tokens, src_lengths=None, **kwargs):
        # Run using pb_utils.InferenceRequest
        inference_request = pb_utils.InferenceRequest(
            model_name='encoder',
            requested_output_names=['OUTPUT__0', 'OUTPUT__1'],
            inputs=[pb_utils.Tensor.from_dlpack('INPUT__0', to_dlpack(src_tokens)), pb_utils.Tensor.from_dlpack('INPUT__1', to_dlpack(src_lengths))]
        )
        inference_response = inference_request.exec()

        if inference_response.has_error():
            raise pb_utils.TritonModelException(
                inference_response.error().message()
            )

        encoder_out = get_torch_from_response(inference_response, 'OUTPUT__0')
        encoder_padding_mask = get_torch_from_response(inference_response, 'OUTPUT__1').to(torch.bool)
        return {
            'encoder_out': [encoder_out.permute(1, 0, 2)],
            'encoder_padding_mask': [encoder_padding_mask],
            'encoder_embedding': [],
            'encoder_states': [],
            'src_tokens': [],
            'src_lengths': []
        }

    def reorder_encoder_out(self, encoder_out: Dict[str, List[Tensor]], new_order):
        """
        Reorder encoder output according to *new_order*.

        Args:
            encoder_out: output from the ``forward()`` method
            new_order (LongTensor): desired order

        Returns:
            *encoder_out* rearranged according to *new_order*
        """
        if len(encoder_out["encoder_out"]) == 0:
            new_encoder_out = []
        else:
            new_encoder_out = [encoder_out["encoder_out"][0].index_select(1, new_order)]
        if len(encoder_out["encoder_padding_mask"]) == 0:
            new_encoder_padding_mask = []
        else:
            new_encoder_padding_mask = [
                encoder_out["encoder_padding_mask"][0].index_select(0, new_order)
            ]
        if len(encoder_out["encoder_embedding"]) == 0:
            new_encoder_embedding = []
        else:
            new_encoder_embedding = [
                encoder_out["encoder_embedding"][0].index_select(0, new_order)
            ]

        if len(encoder_out["src_tokens"]) == 0:
            src_tokens = []
        else:
            src_tokens = [(encoder_out["src_tokens"][0]).index_select(0, new_order)]

        if len(encoder_out["src_lengths"]) == 0:
            src_lengths = []
        else:
            src_lengths = [(encoder_out["src_lengths"][0]).index_select(0, new_order)]

        encoder_states = encoder_out["encoder_states"]
        if len(encoder_states) > 0:
            for idx, state in enumerate(encoder_states):
                encoder_states[idx] = state.index_select(1, new_order)

        return {
            "encoder_out": new_encoder_out,  # T x B x C
            "encoder_padding_mask": new_encoder_padding_mask,  # B x T
            "encoder_embedding": new_encoder_embedding,  # B x T x C
            "encoder_states": encoder_states,  # List[T x B x C]
            "src_tokens": src_tokens,  # B x T
            "src_lengths": src_lengths,  # B x 1
        }


class TritonDecoder(FairseqIncrementalDecoder):
    def __init__(self):
        super().__init__(None)
        self.self_attn_num_heads = 16
        self.self_attn_head_dim = 64
        self.encoder_attn_num_heads = 16
        self.encoder_attn_head_dim = 64
        self.num_decoder_layers = 6

    def forward(self, prev_output_tokens, encoder_out=None, incremental_state={}, **kwargs):
        # Run using pb_utils.InferenceRequest
        batch_size = prev_output_tokens.size(0)
        incremental_state.setdefault('incremental_self_attn_states', torch.zeros((batch_size, 2, self.num_decoder_layers, self.self_attn_num_heads, 1, self.self_attn_head_dim), device='cuda').contiguous())
        incremental_state.setdefault('incremental_encoder_attn_states', torch.zeros((batch_size, 2, self.num_decoder_layers, self.encoder_attn_num_heads, 1, self.encoder_attn_head_dim), device='cuda').contiguous())
        incremental_state.setdefault('incremental_encoder_prev_padding_mask', torch.zeros((batch_size, self.num_decoder_layers, 1), device='cuda').contiguous())

        encoder_padding_mask = encoder_out['encoder_padding_mask'][0].to(torch.int64)
        encoder_out = encoder_out['encoder_out'][0].swapaxes(0, 1).contiguous()


        inference_request = pb_utils.InferenceRequest(
            model_name='decoder',
            requested_output_names=['OUTPUT__0', 'OUTPUT__1', 'OUTPUT__2', 'OUTPUT__3', 'OUTPUT__4'],
            inputs=[
                pb_utils.Tensor.from_dlpack('INPUT__0', to_dlpack(prev_output_tokens.contiguous())),
                pb_utils.Tensor.from_dlpack('INPUT__1', to_dlpack(encoder_out)),
                pb_utils.Tensor.from_dlpack('INPUT__2', to_dlpack(encoder_padding_mask)),
                pb_utils.Tensor.from_dlpack('INPUT__3', to_dlpack(incremental_state['incremental_self_attn_states'].clone(memory_format=torch.contiguous_format))),
                pb_utils.Tensor.from_dlpack('INPUT__4', to_dlpack(incremental_state['incremental_encoder_attn_states'])),
                pb_utils.Tensor.from_dlpack('INPUT__5', to_dlpack(incremental_state['incremental_encoder_prev_padding_mask']))
            ]
        )
        inference_response = inference_request.exec()

        if inference_response.has_error():
            raise pb_utils.TritonModelException(
                inference_response.error().message()
            )

        logits = get_torch_from_response(inference_response, 'OUTPUT__0')
        attn = get_torch_from_response(inference_response, 'OUTPUT__1')
        prev_self_kv = get_torch_from_response(inference_response, 'OUTPUT__2')
        prev_encoder_kv = get_torch_from_response(inference_response, 'OUTPUT__3')
        prev_padding_mask = get_torch_from_response(inference_response, 'OUTPUT__4')

        if prev_output_tokens.shape[1] == 1:
            incremental_state['incremental_self_attn_states'] = prev_self_kv
        else:
            incremental_state['incremental_self_attn_states'] = torch.cat([incremental_state['incremental_self_attn_states'], prev_self_kv], dim=4)
        incremental_state['incremental_encoder_attn_states'] = prev_encoder_kv
        incremental_state['incremental_encoder_prev_padding_mask'] = prev_padding_mask
        return logits, attn

    def reorder_incremental_state(
            self,
            incremental_state: Dict[str, Tensor],
            new_order: Tensor,
    ):
        reorder_incremental_state(incremental_state, new_order)


class TritonTransformerModel(TransformerModel):
    def __init__(self, cfg, encoder, decoder):
        super().__init__(cfg, encoder, decoder)


def get_parameter(parameters: dict, key: str, required: bool = False, default = None) -> Optional[str]:
    parameter = parameters.get(key, None)
    if parameter is None:
        if required:
            raise ValueError(f"{key} is required in config.pbtxt parameters")

        return default

    return parameter['string_value']


def pad_and_stack_sequences(sequences: List[torch.Tensor], pad_value: int) -> torch.Tensor:
    max_len = max([len(sequence) for sequence in sequences])
    padded = torch.full((len(sequences), max_len), pad_value, dtype=torch.long)
    for i, sequence in enumerate(sequences):
        padded[i, :len(sequence)] = sequence

    return padded


class TritonPythonModel:
    """Your Python model must use the same class name. Every Python model
    that is created must have "TritonPythonModel" as the class name.
    """

    def initialize(self, args):
        """`initialize` is called only once when the model is being loaded.
        Implementing `initialize` function is optional. This function allows
        the model to initialize any state associated with this model.

        Parameters
        ----------
        args : dict
          Both keys and values are strings. The dictionary keys and values are:
          * model_config: A JSON string containing the model configuration
          * model_instance_kind: A string containing model instance kind
          * model_instance_device_id: A string containing model instance device ID
          * model_repository: Model repository path
          * model_version: Model version
          * model_name: Model name
        """

        # You must parse model_config. JSON string is not parsed here
        self.model_config = json.loads(args["model_config"])
        parameters = self.model_config.get("parameters", {})
        beam_size = int(get_parameter(parameters, "beam_size", required=False, default=2))
        tgt_dict_path = get_parameter(parameters, "tgt_dict_path", required=True)

        model = TritonTransformerModel(argparse.Namespace(), TritonEncoder(), TritonDecoder())
        tgt_dict = FairseqTask.load_dictionary(tgt_dict_path)

        model = EnsembleModel([model])
        self.sequence_generator = SequenceGenerator(model, tgt_dict, beam_size=beam_size,
                                               max_len_a=1.2,
                                               max_len_b=10,
                                               min_len=1,
                                               normalize_scores=True)



    def execute(self, requests):
        """`execute` MUST be implemented in every Python model. `execute`
        function receives a list of pb_utils.InferenceRequest as the only
        argument. This function is called when an inference request is made
        for this model. Depending on the batching configuration (e.g. Dynamic
        Batching) used, `requests` may contain multiple requests. Every
        Python model, must create one pb_utils.InferenceResponse for every
        pb_utils.InferenceRequest in `requests`. If there is an error, you can
        set the error argument when creating a pb_utils.InferenceResponse

        Parameters
        ----------
        requests : list
          A list of pb_utils.InferenceRequest

        Returns
        -------
        list
          A list of pb_utils.InferenceResponse. The length of this list must
          be the same as `requests`
        """
        # Create stacked tensor from all requests
        src_tokens_list = [get_torch_from_request(request, "src_tokens") for request in requests]
        request_batch_sizes = [src_tokens.shape[0] for src_tokens in src_tokens_list]
        src_tokens = torch.cat(src_tokens_list).cuda()
        src_lengths = torch.cat([get_torch_from_request(request, "src_lengths") for request in requests]).cuda()

        translations = self.sequence_generator({'net_input': {'src_tokens': src_tokens, 'src_lengths': src_lengths}})
        responses = []
        batch_offset = 0
        for request_batch_size in request_batch_sizes:
            translations_tensor = pad_and_stack_sequences(
                [t[0]['tokens'] for t in translations[batch_offset:batch_offset + request_batch_size]], 1)
            batch_offset += request_batch_size
            response = pb_utils.InferenceResponse(
                output_tensors=[pb_utils.Tensor.from_dlpack('translations', to_dlpack(translations_tensor))])
            responses.append(response)

        return responses

    def finalize(self):
        """`finalize` is called only once when the model is being unloaded.
        Implementing `finalize` function is OPTIONAL. This function allows
        the model to perform any necessary clean ups before exit.
        """
        print("Cleaning up...")
