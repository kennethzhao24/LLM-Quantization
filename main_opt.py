import argparse
import torch

from transformers import OPTForCausalLM
from data import get_dataloader
from quantization import RTN, quantize
from utils import find_layers



if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument(
        'model', type=str,
        help='OPT model to load; pass `facebook/opt-X`.'
    )
    parser.add_argument(
        'dataset', type=str, choices=['wikitext2', 'ptb'],
        help='Where to extract calibration data from.'
    )
    parser.add_argument(
        '--gpu', type=int,
        help='Which GPU device to use.'
    )
    parser.add_argument(
        '--wbits', type=int, default=16, choices=[2, 3, 4, 16],
        help='#bits to use for quantization; use 16 for evaluating base model.'
    )
    parser.add_argument(
        '--trits', action='store_true',
        help='Whether to use trits for quantization.'
    )
    parser.add_argument(
        '--sym', action='store_true',
        help='Whether to perform symmetric quantization.'
    )
    parser.add_argument(
        '--save', type=str, default='',
        help='Save quantized checkpoint under this name.'
    )

    args = parser.parse_args()

    # load OPT model
    model = OPTForCausalLM.from_pretrained(args.model, torch_dtype='auto')
    model.eval()

    # set gpu device
    device = torch.device('cuda:{}'.format(str(args.gpu)))

    # load dataset
    testloader = get_dataloader(args.dataset, model=args.model)
    
    # perform round-the-nearest (RTN) quantization
    layers = model.model.decoder.layers

    if args.wbits != 16:
        print('Perform {} bit quantization...'.format(args.wbits))
        for i in range(len(layers)):
            print('Quantizing layer {} to {} bit'.format(i, args.wbits))
            layer = layers[i].to(device)
            subset = find_layers(layer)
            for name in subset:
                quantizer = RTN()
                quantizer.configure(
                    args.wbits, perchannel=True, sym=args.sym, mse=False
                )
                W = subset[name].weight.data
                quantizer.find_params(W, weight=True)
                subset[name].weight.data = quantize(W, 
                                                    quantizer.scale, quantizer.zero, quantizer.maxq
                                                    ).to(next(iter(layer.parameters())).dtype)
            layers[i] = layer.cpu()
            del layer
            torch.cuda.empty_cache()
        print('{} bit quantization completed.'.format(args.wbits))
        model.model.decoder.layers = layers
        if args.save:
            torch.save(model.state_dict(), 'opt_4_bit.pt')
    
    # eval
    seq_len = model.config.max_position_embeddings
    num_samples = testloader['input_ids'].numel() // seq_len
    model = model.to(device)
    losses = []
    print('Start Eval...')
    with torch.no_grad():
        for i in range(num_samples):
            outputs = model(input_ids=testloader['input_ids'][:, (i * 2048):((i + 1) * 2048)].to(device), 
                            labels=testloader['input_ids'][:, (i * 2048):((i + 1) * 2048)].to(device),
                            attention_mask = testloader["attention_mask"][:, (i * 2048):((i + 1) * 2048)].to(device).to(device))
            losses.append(outputs[0])
        loss = torch.mean(torch.stack(losses))
        perplexity = torch.exp(loss)

    print('PPL: {:.2f}'.format(perplexity))