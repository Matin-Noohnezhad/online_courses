<?php
// define variables and set to empty values
$search = $searchTitle = $titleContent = "";
$numOccur = 0;
if ($_SERVER["REQUEST_METHOD"] == "POST") {
    $search = test_input($_POST["search"]);
    $searchTitle = test_input($_POST["searchTitle"]);
    //
    $servername = "localhost";
    $username = "root";
    $password = "";
    $dbname = "assignment_3";
// Create connection
    $conn = new mysqli($servername, $username, $password, $dbname);
    mysqli_set_charset($conn, "utf8");
// Check connection
    if ($conn->connect_error) {
        die("Connection failed: " . $conn->connect_error);
    }

    $sql = "SELECT id,title,content FROM data";
    $result = $conn->query($sql);

    if ($result->num_rows > 0) {
        // output data of each row
        if ($searchTitle == 'عنوان') {
            while ($row = $result->fetch_assoc()) {
//            echo "id: " . $row["id"]. " - Name: " . $row["firstname"]. " " . $row["lastname"]. "<br>";
                if (strpos("aaa" . $row["title"], $search) > 0) {
                    $numOccur++;
                    $titleContent .= "
<div>
                    <form method='post' action='content.php'>
                        <input style='display: none' type='number' value='" . $row['id'] . "' name='idNo'>
                        <input type=\"submit\" class=\"submitLink\" value=\"" . $row['title'] . "\">
                    </form>
</div>               
<div style='height: 100px;'>
                    <p>" . $row["content"] . "</p>
</div>
                    ";
                }
            }
        } else if ($searchTitle == 'عنوان و متن') {
            while ($row = $result->fetch_assoc()) {
//            echo "id: " . $row["id"]. " - Name: " . $row["firstname"]. " " . $row["lastname"]. "<br>";
                if (strpos("aaa" . $row['title'], $search) > 0) {
                    $numOccur++;
                    $titleContent .= "
<div>
                     <form method='post' action='content.php'>
                        <input style='display: none' type='number' value='" . $row['id'] . "' name='idNo'>
                        <input type=\"submit\" class=\"submitLink\" value=\"" . $row['title'] . "\">
                    </form>
                    
</div>               
<div style='height: 100px;'>
                    <p>" . $row['content'] . "</p>
</div>
                    ";
                } else if (strpos("aaa" . $row['content'], $search) > 0) {
                    $numOccur++;
                    $titleContent .= "
<div>
                     <form method='post' action='content.php'>
                        <input style='display: none' type='number' value='" . $row['id'] . "' name='idNo'>
                        <input type=\"submit\" class=\"submitLink\" value=\"" . $row['title'] . "\">
                    </form>
</div>               
<div style='height: 100px;'>
                    <p>" . $row['content'] . "</p>
</div>
                    ";
                }
            }
        }
    } else {
//        echo "0 results";
    }
    $conn->close();


    //
    $string = "<!DOCTYPE html>
<html lang=\"fa\" dir=\"rtl\">
<head>
    <title>نتایج جستجو</title>
    <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">
    <link rel=\"stylesheet\" href=\"../css/style2.css\">
</head>
<body>
<div class=\"row\">
    <div class=\"column\" id=\"d1\">
        " . $titleContent . "
    </div>
    <div class=\"column\" id=\"d2\">
        کلمه جستجو شده:
        " . $search . "
        <br>
        <br>
        <br>
        فیلتر جستجو:
        " . $searchTitle . "
        <br>
        <br>
        <br>
        تعداد نتایج یافت شده:
        " . $numOccur . "
    </div>
</div>

</body>
</html>";

    echo $string;

}
function test_input($data)
{
    $data = trim($data);
    $data = stripslashes($data);
    $data = htmlspecialchars($data);
    return $data;
}

?>