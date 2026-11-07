
const Button1 = document.getElementById("button1");
const Num = document.getElementById("num");
const Ava = document.getElementById("avatar");

//Button1.addEventListener("click", function () {
//    console.log("开始识别");
//});

Button1.addEventListener("click",function(){
    if(Ava.files.length == 0){
        Num.textContent = "请先选择图片";
    }else{
        Num.textContent = "图片已选择，等待识别";
        const data = new FormData();
        const ans = Ava.files[0];
        data.append("image",ans);
        fetch("/predict",{
            method:"POST",
            body: data
        })
            .then(function (response) {
                console.log("状态码：", response.status);
                if(!response.ok){
                    throw new Error("请求失败，状态码：" + response.status);
                }
                return response.json();
            })
            .then(function (result) {
                console.log("服务器返回：", result);
                Num.textContent = "识别结果：" + result.prediction;
            })
            .catch(function (error) {
                console.log("请求出错：", error);
                Num.textContent = "请求失败，请重新输入";
            });
        console.log(data.get("image"));
        }
});

