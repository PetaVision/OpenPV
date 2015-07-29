---
layout: post
title:  "New SystemTest Framework"
date:   2015-5-1 23:01:55
author: Sheng Lundquist
categories: jekyll update
---

Hi all,

I would like to introduce the new testing framework on the repository. Currently, we run runallsystemtests.bash to run all the system tests. In contrast, our system tests are now defined in CMakeLists.txt in the PVSystemTests directory.

There are a couple of reasons why the system tests are being moved to CMakeLists. First, by allowing CMake to handle testing, our testing suite is more portable. Secondly, we can now run all system tests in parallel (with the exception of some tests that depend on the output directory, more on that later), allowing all 224 tests (and growing) to run much faster.

Here's a quick rundown on new system tests.

#Running System Tests:
{% highlight bash %}
cd path/to/PVSystemTests/
ctest[28] [-j <numjobs>]
{% endhighlight %}

The command to run tests is ctest (if you're on the nmc servers, you want to use ctest28, for the same reasons why we run cmake28 on the servers). The -j flag tells how many jobs to run in parallel, mirroring the same -j flag when running make. One thing to note is that the -j flag species how many TESTS (as opposed to processes) to run in parallel, with each MPI test counting as a single test. This means the maximum processes a single test uses can be 4 with 1 thread (more if you're using more than one thread). Keep this in mind when specifying the number of jobs to do in parallel so that you're not overloading the cores.

The default value of number of threads to test with is 1. This can be changed with the *PV_SYSTEM_TEST_THREADS* parameter in ccmake.

You must compile all system tests in "Debug" mode first before running ctest, as that's where it looks for an executable. This requirement is no different from the old system test framework.


Adding a New System Test:
Add the following template to the end of CMakeLists.txt in PVSystemTest

{% highlight cmake %}
set(testBaseName BasicSystemTest) #The base name of the test                                                   
set(testParams BasicSystemTest) #Name of the parameter files in the input directory, leaving out .params
set(testFlags -t ${PV_SYSTEM_TEST_THREADS}) #Extra flags to pass to petavision test
set(testMpi TRUE) #Extra flags to pass to petavision test                
set(testDepends FALSE) #This variable defines if the test is required to run sequentially
AddPVTest(${testBaseName} testParams "${testFlags}" ${testMpi} ${testDepends}) 
{% endhighlight %}

Here's a rundown of each variable.
testBaseName: The base folder name of the system test, ex BasicSystemTest

testParams: A list of parameter files to use for the test. Here's a few ways to specify various setups in our tests:

{% highlight cmake %}
set(testParams myParam1) #One param file in <testBaseDirectory>/input/myParam1.params
set(testParams myParam1 myParam2) #Two param files to test in <testBaseDirectory>/input, one named myParam1.params, other named myParam2.params
set(testParams ) #No parameter files required for the test
{% endhighlight %}

testFlags: Any additional flags required by the run. Note the thread flag that is now required for every run.

testMpi: Determines if the system test runs with 2 MPI processes and 4 MPI processes.

testDepends: Determines if the test should run sequentially with MPI. As an example, BasicSystemTest does not require the output directory for determining if the test passed. Therefore, this flag is set to false so that BasicSystemTest with 1, 2, and 4 MPI processes can all run in parallel. In contrast, CheckpointSystemTest generates an output, runs again from a checkpoint, and compares the new output with the old output. Since the run requires the output of the run, a race condition exists if we run the tests in parallel. Therefore, this flag is set to true to make sure the 3 tests for MPI are running sequentially.

One Final Note:
This new framework required a new method for writing log files. Previously, we depended on bash's output redirection for log file output (runCmd &> mylog.log). Now, we can specify a -l flag (lowercase L) to specify writing to a log file. Example run command:

{% highlight bash %}
Debug/BasicSystemTest -p input/BasicSystemTest -l BasicSystemTest.log -t 1
{% endhighlight %}


Please let me know if you run into problems with the new testing framework.

Sheng


Sheng Lundquist
5/1/15

Hi all,

Through more testing of the testing framework, it looks like there's issues with parallelizing with 1, 2, and 4 processes at the same time. With this in mind, I've taken out the testDepends flag, with it always being true. Secondly, if there are more than one param file to run per test, each parameter file must have a separate output directory, for the same reasons.


