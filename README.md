# deeplearning-network-traffic
Network Traffic Identification with Convolutional Neural Networks - This project aims to implement a new payload-based method to identify network protocol/service using convolutional neural network.

### Network Traffic Dataset
For this study, network traffic was collected during the national CPTC held at RIT in November, 2017. From the collected traffic, 34,929 TCP flows were extracted. These flows contained 24 unique protocol labels, with a fairly unbalanced distribution. The dataset is curated by extracting payload bytes from TCP flows, and the protocol/service labels associated with the flows are detected using a network deep packet inspection tool (nDPI). The following table displays the first few service labels and their associated payload bytes.
<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
      <th>4</th>
      <th>5</th>
      <th>6</th>
      <th>7</th>
      <th>8</th>
      <th>9</th>
      <th>...</th>
      <th>1015</th>
      <th>1016</th>
      <th>1017</th>
      <th>1018</th>
      <th>1019</th>
      <th>1020</th>
      <th>1021</th>
      <th>1022</th>
      <th>1023</th>
      <th>1024</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Google</td>
      <td>22</td>
      <td>3</td>
      <td>1</td>
      <td>2</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>252</td>
      <td>...</td>
      <td>113</td>
      <td>118</td>
      <td>108</td>
      <td>144</td>
      <td>87</td>
      <td>17</td>
      <td>63</td>
      <td>67</td>
      <td>134</td>
      <td>114</td>
    </tr>
    <tr>
      <th>1</th>
      <td>SSL</td>
      <td>22</td>
      <td>3</td>
      <td>3</td>
      <td>0</td>
      <td>57</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>53</td>
      <td>...</td>
      <td>140</td>
      <td>123</td>
      <td>32</td>
      <td>18</td>
      <td>193</td>
      <td>74</td>
      <td>221</td>
      <td>192</td>
      <td>98</td>
      <td>78</td>
    </tr>
    <tr>
      <th>2</th>
      <td>LDAP</td>
      <td>48</td>
      <td>132</td>
      <td>0</td>
      <td>0</td>
      <td>4</td>
      <td>249</td>
      <td>2</td>
      <td>2</td>
      <td>3</td>
      <td>...</td>
      <td>161</td>
      <td>230</td>
      <td>107</td>
      <td>18</td>
      <td>191</td>
      <td>84</td>
      <td>166</td>
      <td>85</td>
      <td>176</td>
      <td>245</td>
    </tr>
    <tr>
      <th>3</th>
      <td>LDAP</td>
      <td>48</td>
      <td>132</td>
      <td>0</td>
      <td>0</td>
      <td>5</td>
      <td>8</td>
      <td>2</td>
      <td>2</td>
      <td>3</td>
      <td>...</td>
      <td>168</td>
      <td>49</td>
      <td>160</td>
      <td>26</td>
      <td>52</td>
      <td>181</td>
      <td>64</td>
      <td>181</td>
      <td>202</td>
      <td>160</td>
    </tr>
    <tr>
      <th>4</th>
      <td>MS_OneDrive</td>
      <td>72</td>
      <td>84</td>
      <td>84</td>
      <td>80</td>
      <td>47</td>
      <td>49</td>
      <td>46</td>
      <td>49</td>
      <td>32</td>
      <td>...</td>
      <td>46</td>
      <td>105</td>
      <td>112</td>
      <td>118</td>
      <td>54</td>
      <td>116</td>
      <td>101</td>
      <td>115</td>
      <td>116</td>
      <td>99</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 1025 columns</p>
</div>

The bar chart shows the most frequent protocols/services and their frequency distribution.
![link](https://github.com/akshitvjain/deeplearning-network-traffic/blob/master/frequency_data/frequency_plot.png)

### Data pipeline for Network Traffic Identification
There are multiple phases through which the payload data needs to pass through before it can be used to train a deep learning model.

![link](https://github.com/akshitvjain/deeplearning-network-traffic/blob/master/images/pipeline.png)

### Results
The table below shows the aggregated performance metrics for the different optimizers used to train the CNN model.

![link](https://github.com/akshitvjain/deeplearning-network-traffic/blob/master/images/results.png)
