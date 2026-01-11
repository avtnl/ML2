import torch
from mltrainer.imagemodels import CNNConfig, CNNblocks

filters_values = list(range(16, 129, 8))  # 16..128 step 8
kernel_values  = [2, 3, 4, 5]
layer_values   = list(range(1, 11))       # 1..10

shapes_count = {}

for f in filters_values:
    for k in kernel_values:
        for L in layer_values:
            config = CNNConfig(
                matrixshape=(28, 28),
                batchsize=32,
                input_channels=1,
                hidden=f,
                kernel_size=k,
                maxpool=3,
                num_layers=L,
                num_classes=10,
            )
            model = CNNblocks(config)

            # --- find the Flatten layer ---
            flatten_module = None
            for m in model.modules():
                if isinstance(m, torch.nn.Flatten):
                    flatten_module = m
                    break
            if flatten_module is None:
                raise RuntimeError("No nn.Flatten found in CNNblocks model.")

            captured = {}

            # --- hook to capture the input to Flatten (i.e. before flatten) ---
            def hook(module, input, output):
                # input is a tuple (tensor,)
                captured["shape"] = input[0].shape  # (B, C, H, W)

            handle = flatten_module.register_forward_hook(hook)

            # --- run a dummy forward pass ---
            x = torch.zeros(1, 1, 28, 28)
            with torch.no_grad():
                _ = model(x)

            handle.remove()

            # just before flatten: (B, C, H, W)
            C, H, W = captured["shape"][1:]
            shape_key = (C, H, W)
            shapes_count[shape_key] = shapes_count.get(shape_key, 0) + 1

# --- report results ---
total = sum(shapes_count.values())
print("Total configs:", total)
for shape, cnt in sorted(shapes_count.items(), key=lambda x: x[0]):
    print(f"{shape} -> {cnt}")
