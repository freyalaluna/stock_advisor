## Setup
All of the underlying Spark and Java dependencies should be preinstalled on the CSU lab machines, so no need to change them. All that needs to be added is the pyspark library.

I used pip to install pyspark locally, as such: 

`pip install pyspark==3.5.2 --user` 

Note: Your pyspark version needs to be 3.5.2, since that's the underlying version of Spark that the lab machines have installed. Trying it with another version of pyspark will lead to errors.

Additional dependencies will be added here (or to a requirements.txt if there are enough) as they come up in our development.
