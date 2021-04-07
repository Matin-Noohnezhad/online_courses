var contents = [
    {
        title: "جذاب ترین خودرو های نمایشگاه SEMA2016",
        summary: "این خودرو های زیبا و گرانقیمت همانطور که حدس میزنید دچار ارزش افزوده بر کالا و مبلغی هنگفت اسیر تبکیت مغلطی در توصیف این موضوع که این ماشین ها همگام با یکدیگر و دست در دست یکدیگر کار را به پیشبرد اهداف هم هستند.",
        picture: "car.jpg",
        category: "خودرو"
    }, {
        title: "سه راه آسان برای شخصی اسزی قفل صفحه ی نمایش در مک",
        summary: "این لپ تاتپ های زیبا و گرانقیمت همانطور که حدس میزنید دچار ارزش افزوده بر کالا و مبلغی هنگفت اسیر تبکیت مغلطی در توصیف این موضوع که این ماشین ها همگام با یکدیگر و دست در دست یکدیگر کار را به پیشبرد اهداف هم هستند.",
        picture: "laptop.png",
        category: "تکنولوژی"
    }
];
func();

function func() {
    var n = contents.length;
    var text = "";
    for (var i = 0; i < n; i++) {
        text += "<div style=\"width: 25%; height:200px;display: inline-block;\">\n" +
            "        <img src='" + contents[i].picture + "' style=\"max-width:100%;max-height: 100%;width:auto;height: auto;\">\n" +
            "    </div>\n" +
            "    <div style=\"width: 65%;height: 200px; margin: auto;display:inline-block;\">\n" +
            "        <h2> " + contents[i].title + "</h2>\n" +
            "        <h4> " + contents[i].category + "</h4>\n" +
            "        <p>" + contents[i].summary + "</p>\n" +
            "    </div>";
        document.getElementById("d1").innerHTML = text;
    }
}

