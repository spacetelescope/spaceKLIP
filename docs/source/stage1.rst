.. _stage1:

Stage 1
=======

The following steps are part of the ``Coron1Pipeline`` custom version of the JWST pipeline's ``Detector1Pipeline``.

Note, the ``coron1pipeline.run_obs()`` function provides a succinct interface to create and invoke Coron1Pipeline on all
observations in a spaceKLIP Database; that's the recommended way to invoke Coron1Pipeline.


group_scale
-----------
Identical to default Detector1Pipeline

dq_init
-----------
Identical to default Detector1Pipeline

saturation
----------
For *MIRI*, identical to default Detector1Pipeline

For *NIRCam*, depending on parameters ``flag_rcsat`` and ``grow_diagonal`` and ``n_pix_grow_sat``, may do some custom flagging of which pixels
should be flagged around saturated pixels.

**Custom step, TBD write more docs for this**

Parameters you may wish to adjust for your data:
 * ``flag_rcsat``. Default = False.
 * ``grow_diagonal``. Default = False.
 * ``n_pix_grow_sat``

ipc
-----------
Identical to default Detector1Pipeline

firstframe, lastframe, reset
----------------------------
(Only for MIRI)
Identical to default Detector1Pipeline

superbias
----------
(Only for NIRCam)
Identical to default Detector1Pipeline

refpix
--------
(Applies to both NIRCam and MIRI, but in different orders with other pipeline steps -- before linearity for NIRCam, after for MIRI??)

**Custom step, TBD  write docs here for this**

For MIRI, runs identical to default Detector1Pipeline.

For NIRCam, for subarray images, runs a custom step to do "psuedo" reference pixel subtraction using pixels around the edge of the subarray.

Parameters you may wish to adjust for your data:
 * ``nlower, nupper, nleft, nright``: number of pixels around each edge to treat as pseudo-refpix. (left and right may not actually do anything, TBC?)


linearity
----------
Identical to default Detector1Pipeline

rscd
-----
(Only for MIRI)
Identical to default Detector1Pipeline

dark_current, charge_migration, jump
-------------------------------------
Identical to default Detector1Pipeline

subtract_ktc
------------
(Only for NIRCam)
Custom step for removal of kTc noise.

**Custom step, TBD  write docs here for this**

subtract_1overf
----------------
(Only for NIRCam)
Custom step for removal of one over f noise, optimized for coronagraphic data using. This can be run either at the groups stage or at the ramp stage (i.e. before
or after ramp fitting)

**Custom step, TBD  write docs here for this**

ramp_fit
--------
Identical to default Detector1Pipeline

apply_rateints_outliers
-----------------------
Custom step to Flag additional outliers by comparing rateints and refit ramp

**Custom step, TBD  write docs here for this**

gain_scale
----------
Identical to default Detector1Pipeline. Applied to both the rate and rateints outputs, as standard.

