from typing import Any

import torch


class Relu(torch.autograd.Function):
    """ReLU activation function.

    This is a direct implementation of Relu using torch's autograd functionality.

    The Forward method computes the actual output where any input values that are
    less than 0 the output is set to 0, any other value is passed through.
    """

    @staticmethod
    def forward(ctx: Any, input: torch.Tensor) -> torch.Tensor:
        """f(x) = 0 if x < 0 else x

        Parameters
        ----------
            ctx (Any): The context object. that allows us to pass saved tensors
                or parameters from the forward pass to the backward pass.
            input (torch.Tensor): The input tensor to the ReLU function.
                This is `batch size x features` in dimension.

        Returns
        -------
            torch.Tensor: The output tensor after applying the ReLU function.
                This will have the same dimensions as the input.
        """
        output = input.clamp(min=0)
        ctx.save_for_backward(input)
        return output

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> torch.Tensor:
        """Calculate the gradient of the input with respect to the output.

        We  are looking to return a gradient for every input into forward aside
        from the context object. So we are looking for df/dx where x is our input.

        The gradient for this is easy. For less than 0 parameters the gradient is 0
        and for any other value the gradient is 1.

        The gradient for the loss function L is passed in as grad_output which is dL/df
        and we want to return dL/dx. By chain rule we apply dL/df df/dx = dL/dx and
        return dL/dx

        Parameters
        ----------
            ctx (Any): The context object. Saved tensors from the forward pass
                exist on this object
            grad_outputs (torch.Tensor): The gradient of the loss function
                with respect to the output, dL/df


        Returns
        -------
            torch.Tensor: The gradient of the input with respect to the output.
                This will have the same dimensions as the input.
        """
        grad_output = grad_output
        (input,) = ctx.saved_tensors

        grad_input = grad_output.clone()
        grad_input[input <= 0] = 0
        return grad_input


class ActFunc(torch.autograd.Function):
    """ActFunc implementation using torch's autograd functionality.

    The ActFunc function is f(x) = x * (e^x) / ( 1 + e^x)
    """

    @staticmethod
    def forward(
        ctx: Any,
        input: torch.Tensor
    ) -> torch.Tensor:
        """Forward pass for the ActFunc function.

        f(x) = x * (e^x) / ( 1 + e^x)

        Parameters
        ----------
            ctx (Any): The context object. that allows us to pass saved tensors
                or parameters from the forward pass to the backward pass.
            input (torch.Tensor): The input tensor to the ActFunc function.
                This is `batch size x features` in dimension.

        Returns
        -------
            torch.Tensor: The output tensor after applying the ActFunc.
                This will have the same dimensions as the input.
        """
        ctx.save_for_backward(input)

        exp = torch.exp(input)
        output = input * exp / (1 + exp)
        return output

    @staticmethod
    def backward(
        ctx: Any,
        grad_output: torch.Tensor
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        """Backward pass for the ActFunc function.

        Parameterssour
        ----------
            ctx (Any): The context object. Saved tensors from the forward pass
                exist on this object
            grad_output (torch.Tensor): The gradient of the loss function
                with respect to the output, dL/df

        Returns
        -------
            tuple[torch.Tensor | None,]:
                The gradient of the input with respect to the loss function (dL/dx)
        """
        (input,) = ctx.saved_tensors

        grad_input = None

        exp = torch.exp(input)
        sig = torch.sigmoid(input)



        if ctx.needs_input_grad[0]:  # dL/dx
            grad_input = (
                (
                    input*sig*(1-sig) + sig
                ) * grad_output
                
            )

        return grad_input
    
    

