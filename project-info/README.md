# Parliamentary Meeting Minutes Analysis

## Objective

Develop a service that allows users to query parliamentary meeting minutes by an entity name, topic or anything else you think is of note. The service should process the provided meeting minutes and return relevant information, in a structured format, given the input query. It is up to you to decide what insights and relationships are meaningful.

An example input could simply be:
```
{"entity": <str_value>}
```
and an example output could be:
```
{"meeting_dates_present": <list_str_value>, "contributions": <dict_values>, "key_events": <dict_values>}
```

This is purely an example to highlight the kind of structured input and outputs expected. You can expand beyond simple entity names as an input, it could be a free text question or an overall sentiment value for instance (that would return any speaker contributions matching sentiment value). It's really up to you to decide how to approach the data and problem!

Please design your solution thoughtfully, applying relevant software design patterns and best practices to ensure maintainability, scalability, and clarity. An ideal solution is straightforward for others to understand, collaborate on, and extend.

**Background**  
You have been provided with a small sample of parliamentary meeting minutes in text format. These documents contain transcripts of various MPs' speeches, which includes their names, timestamps and the content of their contributions. The goal is to create a system that can efficiently process this data and provide meaningful insights through an accessible interface. You are free to use any pre-trained models via API or locally as you see fit, or even train models if you think it is appropriate. 

**Evaluation**  
Your submission will be evaluated against demonstrating applied Machine Learning skills and Engineering competency, particularly around Large Language Models. There are no right or wrong answers we are looking for in terms of output fields, rather we are trying to examine how you approach an open-ended problem and frame it appropriately as an ML Engineer. Although output fields are yours to derive, we are especially interested in efforts to determine behaviour consistency and measure said output quality via an evaluation framework of your design. How you develop and present your solution is the important bit, so take care to think of it holistically rather than focusing on one component. 

**Submission Instructions**   
Please email your submission as a Github link or zipped folder etc, along with any necessary instructions to run your service.



**Good Luck!**  
We look forward to reviewing your solution - if you get stuck or just want to double check something please reach out.


