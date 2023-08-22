import scipy
import torch


def get_FIR_lowpass(order, fc, beta, sr):
    """
    This function designs a FIR low pass filter using the window method. It uses scipy.signal
    Args:
        order(int): order of the filter
        fc (float): cutoff frequency
        sr (float): sampling rate
    Returns:
        B (Tensor): shape(1,1,order) FIR filter coefficients
    """

    B = scipy.signal.firwin(
        numtaps=order, cutoff=fc, width=beta, window="kaiser", fs=sr
    )
    B = torch.FloatTensor(B)
    B = B.unsqueeze(0)
    B = B.unsqueeze(0)
    return B


def apply_low_pass_firwin(y, filter):
    """
    Utility for applying a FIR filter, usinf pytorch conv1d
    Args;
        y (Tensor): shape (B,T) signal to filter
        filter (Tensor): shape (1,1,order) FIR filter coefficients
    Returns:
        y_lpf (Tensor): shape (B,T) filtered signal
    """

    # ii=2
    B = filter.to(y.device)
    # y = y.unsqueeze(1)
    # weight=torch.nn.Parameter(B)

    y_lpf = torch.nn.functional.conv1d(y, B, padding="same")
    # y_lpf = y_lpf.squeeze(1)  # some redundancy here, but its ok
    # y_lpf=y
    return y_lpf
