import torch
import numpy as np
# Training function
def train_epoch(ae, device, dataloader, optimizer):
    ae.train()
    train_loss = 0.0

    for x, _ in dataloader: # x=vertice, _=coupling

        x = x.to(device)
        x_hat = ae(x)
        error = torch.sum((x- x_hat)**2, dim=(1,2,3,4))/torch.sum((x)**2, dim=(1,2,3,4))
        loss = error.sum()
        

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss+=loss.item()

    return train_loss / len(dataloader.dataset)


# Testing function
def test_epoch(ae, device, dataloader):

    ae.eval()
    val_loss = 0.0
    errors=[]
    with torch.no_grad(): # No need to track the gradients
        for x, _ in dataloader:
            x = x.to(device)
            x_hat = ae(x)

            error = torch.sum((x- x_hat)**2, dim=(1,2,3,4))/torch.sum((x)**2, dim=(1,2,3,4))
            loss = error.sum()
            val_loss += loss.item()


            error_cpu = error.to('cpu').detach()  # Move to CPU and detach from the graph
            error_list = error_cpu.tolist()       # Convert to list
            errors.extend(error_list)

    return val_loss / len(dataloader.dataset),errors

def get_output_and_error(ae, device, dataloader):

    ae.eval()
    errors=[]
    x_hats=[]
    with torch.no_grad(): # No need to track the gradients
        for x, _ in dataloader:
            x = x.to(device)
            x_hat = ae(x)
            

            error = torch.sum((x- x_hat)**2, dim=(1,2,3,4))/torch.sum((x)**2, dim=(1,2,3,4))

            error_cpu = error.to('cpu').detach()  # Move to CPU and detach from the graph
            error_list = error_cpu.tolist()       # Convert to list
            errors.extend(error_list)
            x_hats.extend(x_hat.to('cpu').detach().numpy())

    return np.sqrt(errors), x_hats

