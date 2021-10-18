function I = argmin(X, DIM)
%ARGMAX   Find the index of the maximum value of a matrix or vector along
% the dimension specified (DIM).
%
%   Inputs:
%       X                       = Input vector or matrix.
%       DIM                     = Dimension along which to computer maximum argument.
%
%   Outputs:
%       I                       = Index of the min value.
%
% Daniel McDuff, Ethan Blackford, January 2019
% Copyright (c)
% Licensed under the MIT License and the RAIL AI License.

[~,I] = min(X,[],DIM);