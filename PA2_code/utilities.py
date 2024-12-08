import matplotlib.pyplot as plt
import torch
import numpy as np

class Utilities:
    def __init__(self, tokenizer, model):
        self.tokenizer = tokenizer
        self.model = model

    def sanity_check(self, sentence, block_size):
        # Encode the sentence using the tokenizer
        wordids = self.tokenizer.encode(sentence)

        # Prepare the padded input for the model
        padded_sentence = wordids[:block_size] + [0] * (block_size - len(wordids))
        input_tensor = torch.tensor(padded_sentence, dtype=torch.long).unsqueeze(0)

        ##
        attention_mask = (input_tensor != 0).long()

        ## Move tensors to the same device as the model
        device = next(self.model.parameters()).device
        input_tensor = input_tensor.to(device)
        attention_mask = attention_mask.to(device)

        # Display input tensor shape
        print("Input tensor shape:", input_tensor.shape)

        # Process the input tensor through the model
        with torch.no_grad():
            _, attn_maps = self.model(input_tensor, attention_mask) # Ignore the output of the model, and only get the attention maps; make sure your encoder returns the attention maps

        # Display the number of attention maps
        print("Number of attention maps:", len(attn_maps))

        # Visualize and save the attention maps
        for j, attn_map in enumerate(attn_maps):
            # attn_map shape: (batch_size, num_heads, seq_length, seq_length)
            # For visualization, select the first batch and first head
            att_map = attn_map[0, 0].detach().cpu().numpy()  # Select first batch and first head

            # Check if the attention probabilities sum to 1 over rows
            total_prob_over_rows = att_map.sum(axis=-1)

            if np.any(total_prob_over_rows < 0.99) or np.any(total_prob_over_rows > 1.01):
                print("Failed normalization test: probabilities do not sum to 1.0 over rows")
                print("Total probability over rows:", total_prob_over_rows)

            # Create a heatmap of the attention map
            fig, ax = plt.subplots(figsize=(8, 6))
            cax = ax.imshow(att_map, cmap='hot', interpolation='nearest')
            ax.set_title(f"Attention Map {j + 1} (Layer {j+1}, Head 1)")
            fig.colorbar(cax, ax=ax)
            plt.xlabel("Key Positions")
            plt.ylabel("Query Positions")

            # Save the plot
            plt.savefig(f"attention_map_{j + 1}.png")
            plt.close(fig) 

            # # Create a heatmap of the attention map
            # fig, ax = plt.subplots()
            # cax = ax.imshow(att_map, cmap='hot', interpolation='nearest')
            # ax.xaxis.tick_top()  
            # fig.colorbar(cax, ax=ax)  
            # plt.title(f"Attention Map {j + 1}")
            
            # # Save the plot
            # plt.savefig(f"attention_map_{j + 1}.png")
            
            # # Show the plot
            # plt.show()
            


