This is the README file for:
Fergus Babb, N-Body modelling.

First, please ensure you have some form of TeX installation, this 
is required for the plotting preferences I have used. If you do not 
want to/ cannot do this, when running my code, please hash out the two lines:

plt.rcParams.update({"text.usetex": True, "font.family": "serif",
                     "font.serif": ["Computer Modern Serif"]})

and then unhash the line:

plt.rcdefaults()

to reset the preferences. After this has compiled, close all figures and
run again, this should make it such that everything works apart from where
TeX code has been used in captions, you can simply hash these lines out.
--------------------------------------------------------------------------------

For the body of the project:

Validation test:
First see the report PDF, and read Figure 1. After this open the file 'Validation_test.py'
and confirm that gives what you see in Figure 1. An animated plot should also
start, this is identical to 'Validation.gif', so if your computer cannot do animations, simply
hash out lines 248-276. This file should take approximately 2.0s to run, including animation, 
but not including writing the animation to a gif.


Burraus problem:
Return to the report and read Figure 2. After this open 'Burraus_problem.py' and run this code.
When it is calculating the energy differential, the code takes approximately 90s to run, not
not including writing the animation to a gif, whereas when energy is not being calculated it takes 
instead around 5s to run, this is due to the fact that energy is calculated in a nested loop that
I couldn't get around to vectorising, unlike the ODE.
This code should provide an energy plot (if applicable), an animation and the 6 figures as shown in 
2a of the PDF.


Tolerance testing:
Return to the report and read Figure 3. After this open 'LowN_comparisons.py'. This code should 
complete in about 3s, including the energy calculation, but not including writing the animation to
a gif, this is due to much fewer steps than in Burraus Problem. This file should produce two figures, 
one being an animation, the other being the total plot and energy differential plot. Feel free to change 
the tolerances on line 122 (I used things like rtol = rtol*1e6 with MOTH_III to produce Figure 3c) or change
the periodic plot by changing the choice on line 69 to one of the choices from lines 69-77 - Butterfly III, 
Flower in circle, Goggles and Yin Yang III are particularly good to try other than Moth III.


Solar System:
Return again to the report and read Figure 4. Here I just wrote to a file the initial values for the planets
in the solar system, let them evolve and plotted this. Check you agree that the results match Figure 4. This code
takes approximately 10s to run. I have also included a gif for the inner planets.


Single cluster dynamics:
To save computation I decided to only run with 10 bodies as my laptop isnt amazing. See Figure 5 and hopefully you can 
interpret what it shows. If not, and in any case, please view 'Single_cluster.py' and change the value for N and tf.
For 10 bodies this took around 4s to compile, with 20 around 60s and with 60 (the maximim I managed on a previous iteration)
took around 1 hour. The aim was to optimise this further using the particle mesh method but I decided against this to
finish earlier parts of the project to show the progression.


Cluster merger dynamics:
Here I computed the merger for 20 bodies as I was running out of time to submit. Figure 6 shows this plot. Please view
'Starcluster.py' and experiment with N as well as the time period, but note that for later times you may want to extend 'lim'
so it isnt a blank animation for lots of the time.


There were many things I would have liked to extend this to but didnt get the chance to, I hope you have enjoyed playing around/marking
this.











