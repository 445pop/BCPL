# Clustering method for surface defects of steel for periodic defect detection



![teaser](assets/net.png)

In the production of steel products, periodic surface defects may arise due to equipment wear or process fluctuations. Intelligent detection and clustering methods offer a promising approach for identifying such defects, enabling timely localization of fault sources, preventing the spread of quality risks, minimizing economic losses, and advancing predictive maintenance and intelligent manufacturing. However, several challenges exist in practical production scenarios: defects of the same type often exhibit significant variation in texture and morphology; spatial periodicity is difficult to maintain strictly; and clustering algorithms must continuously adapt to newly emerging defect patterns.
	To address these issues, this study proposes a two-stage generalized category discovery framework based on hierarchical distribution alignment regularization (HDAR). The framework leverages existing clustering results as knowledge anchors to implicitly balance the marginal probability distributions between known and unknown classes within each batch, while explicitly promoting prediction consistency within each group. This dual mechanism enables more structured control over the predictive distribution, thereby significantly improving clustering accuracy.Experimental results demonstrate that the proposed method achieves a 4.54% improvement in overall clustering accuracy compared to classical approaches, with a particularly notable gain of 5.87% on unknown categories, highlighting its superior capability in distinguishing novel defect types. This method enhances defect identification and provides timely feedback for equipment maintenance and process optimization, ultimately contributing to improved steel product quality and production stability.

## Running

### Dependencies

```
pip install -r requirements.txt
```

### Config

Set paths to datasets and desired log directories in ```config.py```


### Datasets
We collected five types of periodic defects from steel plants and augmented the dataset with images from the NEU-DET steel surface defect dataset, which contains six defect categories. In total, we constructed a periodic defect dataset comprising 11 classes. Additionally, we selected defect images from the GC10-DET dataset, which includes 10 surface defect categories from other production processes, to conduct cross-scenario validation.

Both datasets are available at the following links:

* [钢材表面周期性缺陷数据集](https://pan.baidu.com/s/1H2qmxsnHj-ZXrbt5xROGDQ?pwd=1234) and [GC10-DET](https://pan.baidu.com/s/1qbm5eYWiWvcAObg5IzD-Ew?pwd=1234)



### Scripts

**Train the model**:

```
bash scripts/run_${DATASET_NAME}.sh
```

## Results
Our results:

<table>
<thead>
<tr>
<th>Source</th>
<th colspan="3">钢材周期性缺陷数据集</th>
<th colspan="3">GC10-DET</th>
</tr>
</thead>
<tbody>
<tr>
<td>Method</td>
<td>All</td>
<td>Known</td>
<td>Unknown</td>
<td>All</td>
<td>Known</td>
<td>Unknown</td>
</tr>
<tr>
<td>DINO+k-means</td>
<td>58.85</td>
<td>73.23</td>
<td>53.28</td>
<td>54.43</td>
<td>72.84</td>
<td>42.41</td>
</tr>
<tr>
<td>DINO+semi-kmeans</td>
<td>67.12</td>
<td>76.22</td>
<td>63.60</td>
<td>63.72</td>
<td>73.25</td>
<td>57.51</td>
</tr>
<tr>
<td>GCD</td>
<td>88.88</td>
<td>98.92</td>
<td>85.00</td>
<td>69.62</td>
<td>95.99</td>
<td>52.42</td>
</tr>
<tr>
<td>DCCL</td>
<td>63.54</td>
<td>83.09</td>
<td>55.98</td>
<td>67.38</td>
<td>74.07</td>
<td>63.02</td>
</tr>
<tr>
<td>GPC</td>
<td>77.11</td>
<td>60.22</td>
<td>83.64</td>
<td>66.25</td>
<td>75.31</td>
<td>60.34</td>
</tr>
<tr>
<td>CMS</td>
<td>73.95</td>
<td>98.28</td>
<td>64.55</td>
<td>74.86</td>
<td>94.14</td>
<td>62.28</td>
</tr>
<tr>
<td>SimGCD</td>
<td>87.01</td>
<td>85.08</td>
<td>87.76</td>		
<td>69.42</td>
<td>89.92</td>
<td>56.04</td>
</tr>
<tr>
<td>PPC-DCR</td>
<td>87.95</td>
<td>97.47</td>
<td>84.27</td>
<td>64.82</td>
<td>81.06</td>
<td>54.22</td>
</tr>
<tr>
<td>LegoGCD</td>
<td>81.69</td>
<td>95.39</td>
<td>76.40</td>
<td>70.51</td>
<td>88.89</td>
<td>58.52</td>
</tr>
<tr>
<td>BPCL-GCD</td>
<td>86.79</td>
<td>91.68</td>
<td>87.27</td>
<td>69.13</td>
<td>88.99</td>
<td>56.17</td>
</tr>
<tr>
<td>SimGCD*</td>
<td>90.34</td>
<td>96.56</td>
<td>87.94</td>
<td>74.25</td>
<td>80.66</td>
<td>70.07</td>
</tr>
<tr>
<td>Happy*</td>
<td>91.83</td>
<td>97.74</td>
<td>89.55</td>	
<td>75.71</td>
<td>87.96</td>
<td>67.72</td>
</tr>
<tr>
<td>HDAR(Ours)</td>
<td>94.88</td>
<td>97.65</td>
<td>93.81</td>	
<td>80.42</td>
<td>92.08</td>
<td>72.82</td>
</tr>
</tbody>
</table>


## Acknowledgements

The codebase is largely built on this repo: https://github.com/CVMI-Lab/SimGCD.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
