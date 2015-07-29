---
layout: post
title:  "Viscosity Momentum Method"
date:   2015-5-21 23:01:55
author: Sheng Lundquist
categories: jekyll update
---

Hi all,

It looks like there was a slight error in the momentum equation I derived for the viscosity method. We have assumed previously that tau\*dwmax with viscosity is equivalent to dwmax without momentum. I have updated MomentumConn to reflect this assumption more accurately. Luckily, the old equation was only off by a factor of 2; a momentum tau of 100 previously is approximately equal to a momentum tau of 200 for the current iteration. In other words, the old code was decaying more slowly than intended. Formally, the equations changed from

*dw = momentumTau \* (dwPrev + inpulse) \* (1-exp(-1/momentumTau)*

to

*dw = dwPrev \* exp(-1/momentumTau) + inpulse*

where

*inpulse = dwMax \* pre \* post* .

