---
layout: post
title:  "Changes to Triggering"
author: Pete Schultz
categories: triggering
---
Several people have said that it would be convenient for a dynamic layer
(HyPerLCALayer, ISTALayer, etc.) to receive triggers.  The idea is that these
layers would still need to update every timestep, except that when a trigger
arrived, they would reset the membrane potential based on the values in a
different layer.

To implement this behavior, there are two new string parameters,
triggerBehavior and triggerResetLayerName.

triggerBehavior is read if triggerLayerName is non-empty.  Currently, there
are two allowed values: "updateOnlyOnTrigger" and "resetStateOnTrigger"
(case sensitive).

triggerResetLayerName is read if triggerBehavior is "resetStateOnTrigger".
It defines the layer to use when resetting membrane potential in response to
the trigger.  If blank, the resetting layer is taken to be the layer named
in triggerLayerName.  The resetting layer must have the same restricted
dimensions as the current layer.

"updateOnlyOnTrigger" implements the behavior that has existed up to now:
on non-triggering timesteps the layer does not receive synaptic input or
update its state.

"resetStateOnTrigger" implements the new behavior:  On non-triggering
timesteps, the layer updates as usual.  However, on non-triggering timesteps,
the layer does not update based on synaptic input. Instead it copies the
membrane potential from the resetting layer and then calls setActivity.

Another trigger-related change: the parameter triggerFlag has been deprecated
as redundant.  Setting triggerLayerName to NULL has the effect of turning
triggering off and setting it to the name of a layer is enough to turn
triggering on.


