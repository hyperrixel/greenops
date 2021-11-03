# greenops
Software to measure the footprints of deep learning models at training, testing and evaluating to reduce energy consumption and carbon footprints.

## How to use

```python

import greenops as go

```

And you can use **greenops** right now.

To begin a new a measuremenet simply type

```python

measure = go.Measure()

```

There are two main approaches to make measurements.

1. You can use ` measure.start() ` and ` measure.stop() ` .
2. You can use ` measure.update() `.

```python

measure.start()

# Your code to measure

measure.stop()

```

or


```python

while condition:

  measure.update()

  # Your code to measure

```

Both ` start() ` , ` stop() ` and  ` upgrade() ` accepts a  ` stage_name ` parameter. You can manage different stages concurrently.

## Go advanced

Instead of ` Measure ` you can use ` advanced.ThreadMeasure ` or ` torchmeasure.TorchMeasure ` as well.

## More details

Coming soon.
