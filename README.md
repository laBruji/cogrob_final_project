# cogrob_final_project

# to run the ros script

so far, mastnu_to_dispatch.py goes from an example mastnu to the dispatchable form for each agent. -> missing the dc_checking part
maybe save the dispatchable form to a json to input into the implementation

probably need to have the dispatcher code with the ros_interaction.py code
The dispatchers run sequentially which is not what we want, they should work in parallel -> use threading to run the online dispatcher in parallel
$ rosrun actionlib_tutorials fibonacci_client.py -> this is how to call a script using ros
