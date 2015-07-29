---
layout: post
title:  "New Blog Documentation"
date:   2015-7-29 5:42:55
author: Sheng Lundquist
categories: jekyll update
---

Hi all,

I've made this blog as a way to keep track of important emails that gets communicated with our pv\_office mailing list. Thanks to Will's email tagging and stitching, I've made a list of emails on this site. There was some discussion as to if we should be using github's issues page to deal with this. Ultimately, the fact that these issues can be closed (which render the "post" invisible) was the deciding factor of not going with this, as the issues page fits more with actual code issues, such as bugs or feature requests. Note that some of the emails on this page actually does belong to issues, but since you can't backdate issues, I decided to add it here instead. Here's everything you need to know about this blog.

This page is generated with Jekyll, a "simple, blog-aware, static sites" generator. Github pages actually uses Jekyll to create it's own pages, making this a natural extension of our github pages. Posts on this page can be tracked via RSS, so you will have to set up your own way to get notifications about blog posts. For example, I created an account on MailChimp that simply checks the feed everyday at 2:00 pm and emails me about it. Currently, my MailChimp is only sending me emails, but if you would like to piggyback off of my system, please let me know and I can add you to the mailing list.

To make a blog post, you will need to checkout the gh-pages branch of OpenPV.

{% highlight bash %}
cd path/to/git/repo/OpenPV
git checkout -b gh-pages origin/gh-pages
cd _posts
{% endhighlight %}

The name of the file must be as follows:

{% highlight text %}
   2015-7-29-my-awesome-blog-post.markdown
{% endhighlight %}

At the beginning of the file, add this header as such:
{% highlight text %}
---
layout: post
title:  "My Awesome Blog Post"
date:   2015-7-29 5:42:55
author: Sheng Lundquist
categories: jekyll update
---
{% endhighlight %}

You then can start writing a blog in markdown. For a markdown reference guide, please follow the links [here](https://guides.github.com/features/mastering-markdown/) and [here](http://jekyllrb.com/docs/templates/).

Sheng
