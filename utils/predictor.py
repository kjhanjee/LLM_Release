# if model_checkpoint_path:
#     if os.path.exists(model_checkpoint_path):
#         checkpoint = torch.load(model_checkpoint_path)
#         model.load_state_dict(checkpoint['model_state_dict'])

# sequences = tokenizer.encode("Hey There!").ids
# sequences2 = sequences
# for _ in range(0,2049 - len(sequences2)):
#     sequences2.append(24)
    
# input_tensor = torch.zeros((1, 2048), dtype=torch.long)
# mask_tensor = torch.zeros((1, 2048), dtype=torch.long)

# input_tensor = torch.from_numpy(np.array(sequences2[:-1])).reshape((1,2048))


# mask_tensor = torch.from_numpy(np.array(sequences2[:-1]))
# if len((mask_tensor==24).nonzero(as_tuple=True)[0].size()) == 0:
#         mask_tensor = mask_tensor[(mask_tensor==24).nonzero(as_tuple=True)[0]:] = 0
#         mask_tensor = mask_tensor[:(mask_tensor==24).nonzero(as_tuple=True)[0]] = 1
# else:
#         mask_tensor = torch.from_numpy(np.array([1 for item in range(0,2048)]))



# for i in range(0,10):
    
#     out = torch.softmax(model.predict(input_tensor,mask_tensor),dim=-1)
#     for tensor in out:
#         output = []
#         for item in tensor[len(sequences)-1:]:
#             out_token = int(np.argmax(item.detach().cpu().numpy()))
#             if not out_token == 24:
#                 output.append(out_token)
#                 sequences.append(out_token)

#         if len(sequences) < 2048:
#             sequences2 = sequences

#             for _ in range(0,2049 - len(sequences2)):
#                 sequences2.append(24)
#         else:
#             sequences2 = sequences[len(sequences)-2048:]
#         input_tensor = torch.zeros((1, 2048), dtype=torch.long)
#         mask_tensor = torch.zeros((1, 2048), dtype=torch.long)

#         input_tensor = torch.from_numpy(np.array(sequences2)).reshape((1,2048))

#         mask_tensor = torch.from_numpy(np.array(sequences2)).reshape((1,2048))

# print(tokenizer.decode(sequences))


#     #print(tokenizer.decode([np.argmax(token_probs) for token_probs in out],skip_special_tokens = False))