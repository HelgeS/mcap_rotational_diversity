# Multi-Cycle Assignment Problems with Rotational Diversity
Source code for paper "Multi-Cycle Assignment Problems with Rotational Diversity":

> In multi-cycle assignment problems with rotational diversity, a set of tasks has to be repeatedly assigned to a set of agents. Over multiple cycles, the goal is to achieve a high diversity of assignments from tasks to agents. At the same time, the assignments' profit has to be maximized in each cycle. Due to changing availability of tasks and agents, planning ahead is infeasible and each cycle is an independent assignment problem but influenced by previous choices. We approach the multi-cycle assignment problem as a two-part problem: Profit maximization and rotation are combined into one objective value, and then solved as a General Assignment Problem. Rotational diversity is maintained with a single execution of the costly assignment model. Our simple, yet effective method is applicable to different domains and applications. Experiments show the applicability on a multi-cycle variant of the multiple knapsack problem and a real-world case study on the test case selection and assignment problem, an example from the software engineering domain, where test cases have to be distributed over compatible test machines. 

## How to use

To run one of the rotational diversity strategies:
`python main.py <instance> <strategy> <options>`

```
$ python main.py --help
usage: main.py [-h] [-p {max_assignment,mulknap}] [-t THRESHOLD]
               [--limit-assignments] [--timeout TIMEOUT] [-o OUTPUT]
               [--ind-weights]
               instance
               {profit,affinity,switch,productcomb,wpp,negotiation,exchange}

positional arguments:
  instance
  {profit,affinity,switch,productcomb,wpp,negotiation,exchange}

optional arguments:
  -h, --help            show this help message and exit
  -p {max_assignment,mulknap}, --problem {max_assignment,mulknap}
  -t THRESHOLD, --threshold THRESHOLD
                        Affinity Pressure Threshold (used with strategies
                        adaptive and switch)
  --limit-assignments   Limited assignment, disallow prev. agents
  --timeout TIMEOUT     CP solver timeout (in s)
  -o OUTPUT, --output OUTPUT
  --ind-weights         Use Individual Weights for WPP strategy
```

## Publications

This software has been used in the paper "Multi-Cycle Assignment Problems with Rotational Diversity" ([Preprint](https://arxiv.org/abs/1811.03496)):
Spieker, H., Gotlieb, A., & Mossige, M. (2019). Rotational Diversity in Multi-Cycle Assignment Problems. In AAAI-19.

## License

All of our implementation is licensed under [MIT License](LICENSE). 
[mulknap](mulknap) is licensed as described on [the author's homepage](http://hjemmesider.diku.dk/~pisinger/codes.html).
