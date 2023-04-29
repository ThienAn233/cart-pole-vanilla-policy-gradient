# cart-pole-vanilla-policy-gradient
Using policy gradient algorithm to interact with cart-pole environment
## Result: Without a baseline 
The agent tends to dirft to the right. High variance return.


https://user-images.githubusercontent.com/90390412/235304591-f64696ca-fbae-4c6a-ae44-4ed66cec33e1.mp4


## Result: With a baseline $V(s_t)$ update after every two policy step.
With the same amount of experience, the agent learn not to go too far from the center and achieve maximum reward

https://user-images.githubusercontent.com/90390412/235304878-24d4d650-2e43-4c94-b865-8ce91b4781b7.mp4

