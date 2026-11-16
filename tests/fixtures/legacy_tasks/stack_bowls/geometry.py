"""Vectorized, color-independent nesting checks; no simulator dependencies."""
import math
import torch


def nesting_metrics(positions, quaternions, velocities, settings):
    order = torch.argsort(positions[:,:,2], dim=1)
    sorted_positions = torch.gather(positions, 1, order[:,:,None].expand(-1,-1,3))
    difference = sorted_positions[:,1:]-sorted_positions[:,:-1]
    paired = (torch.linalg.vector_norm(difference[:,:,:2], dim=-1) <= settings['xy_tolerance']) & (
        difference[:,:,2] >= settings['nested_height_min']) & (
        difference[:,:,2] <= settings['nested_height_max'])
    # Compare neighboring bowl axes, allowing cumulative lean in a nested
    # stack. The bottom bowl must still stand upright relative to gravity.
    x,y,z,w = quaternions.unbind(-1)
    up = torch.stack((2*(x*z+y*w),2*(y*z-x*w),1-2*(x*x+y*y)),dim=-1)
    sorted_up = torch.gather(up,1,order[:,:,None].expand(-1,-1,3))
    support_aligned = (sorted_up[:,1:]*sorted_up[:,:-1]).sum(-1) >= math.cos(
        math.radians(settings.get('max_support_tilt_degrees',15.)))
    paired &= support_aligned
    upright = up[:,:,2] >= math.cos(math.radians(settings['max_tilt_degrees']))
    bottom_valid = sorted_up[:,0,2] >= math.cos(math.radians(settings.get('max_bottom_tilt_degrees',12.)))
    upright &= bottom_valid[:,None]
    stable = (torch.linalg.vector_norm(velocities[:,:,:3], dim=-1) <= settings['max_speed']) & (
        torch.linalg.vector_norm(velocities[:,:,3:], dim=-1) <= settings['max_angular_speed'])
    return paired, upright, stable


def any_nested_pair(positions, settings, quaternions=None):
    delta = positions[:,:,None,:]-positions[:,None,:,:]
    mask = (torch.linalg.vector_norm(delta[:,:,:,:2],dim=-1) <= settings['xy_tolerance']) & (
        delta[:,:,:,2] >= settings['nested_height_min']) & (delta[:,:,:,2] <= settings['nested_height_max'])
    if quaternions is not None:
        x,y,z,w = quaternions.unbind(-1)
        up = torch.stack((2*(x*z+y*w),2*(y*z-x*w),1-2*(x*x+y*y)),dim=-1)
        aligned = (up[:,:,None,:]*up[:,None,:,:]).sum(-1) >= math.cos(
            math.radians(settings.get('max_support_tilt_degrees',15.)))
        mask &= aligned
    return mask.any(2).any(1)
