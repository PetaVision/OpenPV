---
layout: post
title:  "Timer Pie, My Favorite Type of Pie"
date:   2015-6-9 23:01:55
author: Brian Broom-Peltz
categories: jekyll update
---

Hey everyone,

PetaVision has a new tool for looking at your timing information that I want to point y'all too. It is currently at two step process to use it but it will give you a colorful interactive pie chart:

HOW TO USE:

Step 1. Navigate to your Checkpoint directory

Step 2. Make a .csv version of your timers.txt 
      
{% highlight bash %}
python ~/workspace/PetaVision/plab/timer_txt_to_csv.py
{% endhighlight %}

Step 3. scp the *timers.csv* to your local machines copy of *Petavision/analysis/timing_info/timing_pie/*

Step 4. Open *petavision_timing_pie.html* in Firefox

You can click on the boxes on the right and that timer will disappear from the pie chart.  
If you MouseOver the pie chart, the name of the element will pop-up with the time and a percentage of that time and all of the other active timers.
The boxes are colored with a repeated scheme that matches the columns to make it easier to find them.

